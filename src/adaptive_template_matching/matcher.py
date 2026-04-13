import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import random

"""
Written by Grange Simpson and Ivan Khimach
Version: 2026.04.11

References:
    NumPy Developers. `numpy.lib.stride_tricks.sliding_window_view`.
    NumPy Developers. `numpy.corrcoef`.
"""

class AdaptiveTemplateMatching:
    """
    Sliding-window template matcher with multi-stage gating and optional merging.

    Pipeline (per scan)
    -------------------
    1) Build sliding windows and apply absolute-amplitude and min-std gates.
    2) Z-score each window to align spaces with the z-scored template.
    3) Compute SAD (−Σ|Δ|) in z-space (diagnostics); optional low-amplitude gate.
    4) For surviving candidates, compute Pearson r; require r ≥ `threshold`.
    5) Fit y = a*T + b and compute relative RMS error and flat-region RMS gate.
    6) Cluster-merge accepted indices within ±`cluster_radius` using metric
       {'r','sad','sad_shifted'} to reduce near-duplicates.
    7) Record SAD traces to `sad_history` for QC.

    Attributes
    ----------
    template : np.ndarray (z-scored)
    win : int
        Template window length.
    sad_history : list[np.ndarray]
        SAD traces per scan for diagnostics.

    Update rule
    -----------
    `_update_template` averages ALL accepted z-scored windows, then z-scores again.

    References:
        NumPy Developers. `numpy.lib.stride_tricks.sliding_window_view`.
        Pearson, K. (1895). Notes on regression and inheritance in the case of
        two parents.
    """
    def __init__(self, cp_inds: list = [145, 195], template_angls: list = [80, 88], 
                template_len: int = 200, baseline: float = 0.0, template_scaler: float = 0.12, 
                reflect: bool = False):
        """Convert the given angle in degrees to a slope. A slope < 90° will be positive, a slope > 90° will be negative.

        Args:
            cp_inds: Indices for points within the template where the slope changes based on the given angles.
            template_angls: Angles to be used in creating the change in slope between the given change points.
            template_len: Wanted length of the template.
            baseline: Starting value used when constructing the initial template.
            template_scaler: Scale factor applied to the normalized template shape.
            reflect: Reflect the template about the y-axis, such as for marking toe-off rather than heel-strike.
        Returns:
            Slope of the line needed to create the line at the given angle.
        """
        self.template_scaler = template_scaler
        self.template = self._make_dual_angle_template(template_angls, cp_inds, template_len, 
                                                baseline)
        
        # Flipping the template if wanted, say for marking toe-off rather than heel-strike.
        if reflect:
            self.template = self.template[::-1]

        # Save of starting template shape to be used when _resetting the template between datasets.
        self.template_save = self.template.copy()

        # Normalizing template shape and initializing other global variables.
        self._reset()
        self.found_indices_scores = np.array([])

    # ===== Template construction ======================================================
    def _angle_to_slope(self, angle_deg):
        """Convert the given angle in degrees to a slope by calculating the ratio of the vertical change to the horizontal change. 
           Utilizing tan(θ) = sin(θ) / cos(θ) = ∆y / ∆x. Angles < 90° will be a positive slope, angles > 90° will be a negative slope.

        Args:
            angle_deg: Input angle in degress to be used for calculating the wanted slope.
        Returns:
            Slope of the line needed to create the line at the given angle.
        """
        return np.tan(np.deg2rad(angle_deg))
    
    def _make_dual_angle_template(self, angle_degr_arr: list, 
                                change_points_arr: list,
                                template_length: int=200,
                                baseline: float = 0.0):
        """Create the piecwise linear function that approximates the wanted template shape based off the given change points and angles.

        Args:
            angle_degr_arr: Array of the wanted slope changes within the template.
            change_points_arr: Array of the indices for a change in slope to occur within the template.
            template_length: Max length of the template.
            baseline: Starting values for the template, i.e. if set to zero, then the beginning values will be zero before the given slopes and change points.
        Returns:
            Template constructed from the given slopes, changepoints, and length.
        """

        for cp in change_points_arr:
            if cp > template_length:
                raise ValueError(f"Change point {cp} not in range (0, {template_length}]")
                return

        # 1. Convert degrees to slopes and apply the direction multiplier
        slopes = [self._angle_to_slope(a) for a in angle_degr_arr]
        
        t = np.zeros(template_length)
        current_t = baseline
        
        # 2. Add total_length to indices to define the final segment's boundary
        break_points = list(change_points_arr) + [template_length]
        
        for i in range(len(slopes)):
            start_idx = break_points[i]
            end_idx = break_points[i+1]
            
            segment_len = end_idx - start_idx
            segment_x = np.arange(segment_len)
            # Calculate y-values starting from where the last segment left off
            segment_t = current_t + (slopes[i] * segment_x)
            
            t[start_idx:end_idx] = segment_t
            
            current_t += slopes[i] * segment_len
                
        return t

    def _corrcoef_safe(self, template: list, data_window: list):
        """Calculating pearson-r correlation coefficient between the template and a data_window to find the data 
            window with the highest correlation to the template.

        Args:
            template: Current template
            data_window: Window of data to correlate the template to
        Returns:
            0 when either vector has zero variance, else returns correlation coefficient between a and b.

        References:
            Pearson, K. (1895). Notes on regression and inheritance in the case
            of two parents.
            NumPy Developers. `numpy.corrcoef`.
        """
        return 0.0 if template.std() == 0 or data_window.std() == 0 else float(np.corrcoef(template, data_window)[0, 1])

    def _rolling_window_correlation(self, template: list, windowed_data: list):
        """ Matrix math for Pearson Correlation Coefficient (r) calculated across the windowed data

        Args:
            template: current template shape in a list.
            windowed_data: input data that has been 
        Returns:
            0 when either vector has zero variance, else returns correlation coefficient between a and b.

        References:
            Pearson, K. (1895). Notes on regression and inheritance in the case
            of two parents.
        """
        # Center the data (Subtract the mean)
        # We subtract the mean of each window and the mean of the template
        windows_centered = windowed_data - np.mean(windowed_data, axis=1, keepdims=True)
        template_centered = template - np.mean(template)
        
        # Calculate the numerator (Dot product along the window axis)
        numerator = np.sum(windows_centered * template_centered, axis=1)
        
        # Calculate the denominator (Product of the norms)
        windows_norm = np.sqrt(np.sum(windows_centered**2, axis=1))
        template_norm = np.sqrt(np.sum(template_centered**2))
        
        return numerator / (windows_norm * template_norm)

    def _matrix_accepted_inds_rms_calculations(self, pearson_corr_arr: np.ndarray, norm_data: np.ndarray, 
                                                idx_cand:np.ndarray, rel_err: float, threshold: float, first_half_err_thresh: float):
        """ RMS calculations between template and data around accepted indices after _rolling_window_correlation 

        Args:
            pearson_corr_arr: Output from _rolling_window_correlation.
            norm_data: Normalized windowed data.
            idx_cand: Candidates indices for highest match with template after passing filters.
            threshold: threshold for pearson correlation value.
            first_half_err_thresh: threshold for pearson rms correlation of first half of the template.
        Returns:
            accepted, scores: The accepted indices after rms calculations against the template, and rms scores.

        References:
            Strang, G. (2016). Introduction to Linear Algebra.
            NumPy Developers. `numpy.linalg.pinv`.
        """
        # Filter pearson_corr_arr and idx_cand based on the threshold
        mask_initial = pearson_corr_arr >= threshold
        idx_filtered = idx_cand[mask_initial]
        scores_filtered = pearson_corr_arr[mask_initial]

        # Extract only the windows that passed the correlation check
        # Shape: (N_candidates, W)
        segs = norm_data[idx_filtered]
        N, W = segs.shape

        if N == 0:
            return [], []

        # Vectorized Least Squares Solve 
        # We want to solve segs = a * template + b
        # Construct Design Matrix A: [template_column, ones_column]
        template = self.template.flatten()
        A = np.column_stack([template, np.ones(W)]) # Shape: (W, 2)

        # Pre-calculate the Pseudoinverse of A once
        # Result coeffs shape: (2, N)
        A_pinv = np.linalg.pinv(A)
        coeffs = A_pinv @ segs.T 

        a = coeffs[0, :, None] # Slopes, reshaped to (N, 1) for broadcasting
        b = coeffs[1, :, None] # Intercepts, reshaped to (N, 1) for broadcasting

        # Calculate Fits and Errors (N, W)
        # Broadcast: (N, 1) * (1, W) + (N, 1) -> (N, W)
        seg_fit = (a * template) + b
        diff_sq = (segs - seg_fit)**2

        # Relative RMS Calculation
        # Numerator: RMS of residuals
        rms_num = np.sqrt(diff_sq.mean(axis=1))
        # Denominator: Std of the fit (RMS of mean-centered fit)
        rms_den = np.std(seg_fit, axis=1) + 1e-12
        rel_rms_vals = rms_num / rms_den

        # Check first half of the window against template
        half_w = W // 2
        rms_flat_vals = np.sqrt(diff_sq[:, :half_w].mean(axis=1))

        final_keep = (rel_rms_vals <= rel_err) & (rms_flat_vals <= first_half_err_thresh)

        accepted = idx_filtered[final_keep].tolist()
        scores = scores_filtered[final_keep].tolist()

        accepted = np.asarray(accepted, int)
        scores   = np.asarray(scores,   float)

        return accepted, scores

    def _min_max_norm(self, x):
        """ Min–max normalization (0–1).

        Args:
            x: Input data to normalize.
        Returns:
            Returns data normalized between 0 and 1.

        References:
            Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and
            Techniques.
        """
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-12)

    def _reset(self):
        """ _Reset the template shape to the initial basic shape if starting on a new dataset.

        Args:
            None.

        Returns:
            None.

        References:
            Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and
            Techniques.
        """
        self.template = self._min_max_norm(self.template_save.astype(float))
        self.template = self.template * self.template_scaler
        self.template_history = [self.template.copy()]
        self.win = len(self.template)
        self.sad_history = []      
        self.last_nsad = None   # last backend z-space SAD

    # --- visualization-only helpers (do NOT affect backend / gating) -------------------------
    def _moving_average_1d(self, y: np.ndarray, w: int) -> np.ndarray:
        """ Odd-length moving average for smoother plots.

        Args:
            y: Data to smooth.
            w: Window size for smoothing.
            
        Returns:  
            Smoothed data.

        References:
            Oppenheim, A. V., and Schafer, R. W. (2009). Discrete-Time Signal
            Processing.
        """
        L = len(y)
        w = max(3, min(int(w), L - 1))
        if w % 2 == 0:
            w += 1
        k = np.ones(w, dtype=float) / w
        return np.convolve(np.asarray(y, float), k, mode="same")

    def plot_template(self, title="Template", raw = False):
        """ Template shape plotting function for debugging purposes.

        Args:
            title: Title for the pop up plot.
            raw: Bool for plot unnormalized or normalized template shape.

        References:
            Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
        """
        data = self.template if raw else self._min_max_norm(self.template)
        plt.figure(figsize=(8, 3))
        plt.plot(data, c="purple")
        plt.title(title + (" (raw z-scored)" if raw else ""))
        plt.grid(); plt.tight_layout(); plt.show()

    def _plot_sad_legacy_visual(self, sad_vis: np.ndarray, title="−Σ|Δ|  (no shift)",
                                figsize: tuple=(14, 2.8)):
        """ Plotting the sum absolute difference signal.

        Args:
            sad_vis: Sum absolute difference signal to plot.
            title: Title for the output plot.
            figsize: tuple of output plot length and height.

        References:
            Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
        """
        #y = np.minimum(np.asarray(sad_vis, float), 0.0)
        y = np.asarray(sad_vis, float)
        y_min = float(np.min(y)); y_max = float(np.max(y))
        pad_bottom = 0.05 * (abs(y_max - y_min)) if (y_max > y_min) else 1.0
        pad_top = 0.05 * (abs(y_min - y_max)) if (y_min < y_max) else 1.0

        plt.figure(figsize=figsize)
        plt.plot(y, lw=1.25, label="−Σ|Δ| (legacy, smoothed)")
        plt.axhline(-0.5, c="red", ls="--", lw=1.0, label="threshold = -0.5")
        #plt.ylim(y_min - pad_bottom, y_max + 0.02 * max(1.0, abs(y_max)))
        plt.ylim(y_min - pad_bottom, y_max + pad_top)
        plt.title(title)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

    def _debug_plots(self, signal_data: np.ndarray, mark_data: np.ndarray, horiz_axis_line_inds: list, 
                    vert_axis_line_inds: list, title: str, label: object, plot_option: str):
        """Calculating pearson-r correlation coefficient between the template and a data_window to find the data 
            window with the highest correlation to the template.

        Args:
            signal_data: Input signal to plot.
            mark_data: template best match marking indices.
            horiz_axis_line_inds: List of horizontal values to place on the plot at specified indices.
            vert_axis_line_inds: List of vertical values to place on the plot at specified indices.
            plot_option: Gating for plots, can either be "sad_legacy", "matches", or "template_composite"

        References:
            Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
        """
        # Debug: panel 1
        if plot_option == "sad_legacy":
            plt.figure(figsize=(12, 2.2))
            plt.plot(signal_data, lw=.8, c="k", alpha=.7, label="signal")
            for thresh in horiz_axis_line_inds: plt.axhline(thresh, c="purple", ls="--", lw=1, label=f"{int(label*100)}th pct")
            xs = mark_data; ys = signal_data[mark_data] if vert_axis_line_inds.size else np.array([])
            if xs.size:
                plt.scatter(xs, ys, s=3, c="purple", alpha=.6, label="candidates (low-amp)")
            plt.title(title)
            plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

        if plot_option == "matches":
            w = self.win
            plt.figure(figsize=(12, 4))
            plt.plot(self._min_max_norm(signal_data), c="black", alpha=.8)
            for i in vert_axis_line_inds: plt.axvspan(i, i+w-1, color="royalblue", alpha=.25)
            plt.title(f"{title}  final matches (n={len(mark_data)})")
            plt.grid(); plt.tight_layout(); plt.show()

        if plot_option == "template_composite":
            plt.figure(figsize=(10, 5))
            draw_segs = signal_data if len(signal_data) <= 200 else signal_data[:200]
            for s in draw_segs:
                plt.plot(s, c="grey", alpha=0.25)
            plt.plot(self._min_max_norm(self.template), c="blue", lw=2, label=f"{label}")
            plt.title(f"{title}")
            plt.grid(); plt.legend(); plt.tight_layout(); plt.show()


    # --- Core template matching algorithm functions-------------------------
    def _scan_matches(self, data,
                     threshold,
                     amp_max,
                     rel_err,
                     flat_err_thresh=0.15,
                     pos_shift=20,
                     sad_thresh=0,
                     min_std=1e-3,
                     # low-amplitude gating
                     enforce_low_amp=True,
                     low_amp_quantile=0.10,
                     low_amp_cover=0.60,
                     # clustering controls
                     cluster_metric="sad",      # {"r","sad","sad_shifted"}
                     cluster_radius=None,       # None → defaults to win
                     show_debug=True,
                     debug_verbose=False):
        """Scan the signal for windows that match the current template.

        Args:
            data: Input signal to scan.
            threshold: Minimum Pearson correlation required for a candidate.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            rel_err: Maximum allowed relative RMS error after template fitting.
            flat_err_thresh: Maximum allowed RMS error in the first half of the window.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches. Must be `"r"`, `"sad"`, or `"sad_shifted"`.
            cluster_radius: Maximum distance between nearby matches before clustering; defaults to the template length.
            show_debug: Whether to show debugging plots.
            debug_verbose: Whether to print verbose scan diagnostics.

        Returns:
            Accepted match indices and their corresponding correlation scores.

        References:
            NumPy Developers. `numpy.lib.stride_tricks.sliding_window_view`.
            Pearson, K. (1895). Notes on regression and inheritance in the case
            of two parents.
        """

        w, N = self.win, len(data)
        if N < w + 1:
            return np.empty(0, int), np.empty(0, float)
        if cluster_radius is None:
            cluster_radius = w

        # Sliding windows (N-w+1, w)
        win_mat = sliding_window_view(data, w)

        # Absolute amplitude gate
        amp_abs_max = np.max(np.abs(win_mat), axis=1)
        amp_ok = amp_abs_max <= amp_max

        # Per-window z-score to match template space
        #win_mean = win_mat.mean(axis=1, keepdims=True)
        win_std  = win_mat.std(axis=1, keepdims=True) + 1e-12
        std_ok   = (win_std.squeeze() >= min_std)
        #win_norm = (win_mat - win_mean) / win_std
        #win_norm = self._zscore(win_mat)
        win_norm = self._min_max_norm(win_mat)

        #print(self.template)
        nsad = -np.sum(np.abs(win_norm - self.template), axis=1) + pos_shift
        #nsad = -np.sum(np.abs(win_mat - self.template), axis=1)
        self.last_nsad = nsad
        self.sad_history.append(nsad.copy())

        # Optional low-amp gate
        if enforce_low_amp:
            q_low   = np.quantile(data, low_amp_quantile)
            frac_lo = (win_mat <= q_low).mean(axis=1)
            low_ok  = frac_lo >= float(low_amp_cover)
        else:
            low_ok = np.ones(len(nsad), dtype=bool)

        # Candidates by inexpensive gates
        idx_cand = np.nonzero((nsad > sad_thresh) & amp_ok & std_ok & low_ok)[0]

        # VISUAL SAD : raw windows vs template
        #sad_vis = -np.sum(np.abs(win_mat - self.template), axis=1)
        sad_vis = nsad.copy()
        #sad_vis = np.minimum(sad_vis, 0.0)

        pearson_corr_arr = self._rolling_window_correlation(self.template, win_norm[idx_cand])

        accepted, scores = self._matrix_accepted_inds_rms_calculations(pearson_corr_arr = pearson_corr_arr, norm_data = win_norm, 
                                                                    idx_cand = idx_cand, rel_err= rel_err, threshold = threshold, first_half_err_thresh = flat_err_thresh)
        # Debug: panel 1
        if show_debug:
            self._plot_sad_legacy_visual(sad_vis, title="−Σ|Δ|  (no shift)")

            if enforce_low_amp:
                self._debug_plots(signal_data = data, mark_data = accepted, horiz_axis_line_inds = [],
                                vert_axis_line_inds = idx_cand, title = "Accepted After RMS Calculations", label = low_amp_quantile, 
                                plot_option = "sad_legacy")

        # Cluster merge 
        if accepted.size > 1:
            order    = np.argsort(accepted)
            acc_sort = accepted[order]
            r_sort   = scores[order]

            if cluster_metric == "r":
                metric_seq = r_sort
            elif cluster_metric == "sad":
                metric_seq = nsad[acc_sort]
            elif cluster_metric == "sad_shifted":
                tmp = nsad.copy()
                if acc_sort.size:
                    tmp[acc_sort] = tmp[acc_sort] #+ pos_shift
                metric_seq = tmp[acc_sort]
            else:
                raise ValueError("cluster_metric must be 'r', 'sad', or 'sad_shifted'.")

            keep = []
            k = 0
            while k < len(acc_sort):
                cluster = [k]
                while k + 1 < len(acc_sort) and acc_sort[k + 1] - acc_sort[k] <= cluster_radius:
                    k += 1; cluster.append(k)
                best_local = cluster[np.argmax(metric_seq[cluster])]
                keep.append(best_local)
                k += 1
            
            # Centering Accepted
            accepted = acc_sort[keep] + (len(self.template) / 2)
            scores   = r_sort[keep]

        if debug_verbose:
            print(f"windows:{N-w+1} | cand:{idx_cand.size} | final:{accepted.size} | cluster_metric={cluster_metric}")

        return accepted, scores

    def _update_template(self, data: np.ndarray, match_idx: np.ndarray, show_debug: bool=True):
        """ Update the template shape using data from accepted indices.

        Args:
            data: Input signal to extract signal from to update indices
            match_idx: indices for best match indices in the signal and template.
            show_debug: Popup plots for debugging purposes.

        References:
            Han, J., Kamber, M., and Pei, J. (2011). Data Mining: Concepts and
            Techniques.
        """
        if match_idx.size == 0:
            print("⚠️  No matches – template unchanged.")
            return
        w = self.win
        segs = [self._min_max_norm(data[int(i - w / 2):int(i + w / 2)]) for i in match_idx]
        self.template = np.mean(segs, axis = 0)
        self.template = self.template * self.template_scaler
        self.template_history.append(self.template.copy())

        if show_debug:
            self._debug_plots(signal_data = segs, mark_data = np.array([]), horiz_axis_line_inds = [], vert_axis_line_inds = [],
                            title = f"Template updated from all accepted segments (n={len(segs)})", label = "New template (ALL)",
                            plot_option = "template_composite")


    def _final_template_match_and_plot(self, name, data,
                                  threshold, amp_max,
                                  pos_shift, sad_thresh,
                                  min_std=1e-3,
                                  rel_err=0.20,
                                  enforce_low_amp=True,
                                  low_amp_quantile=0.10,
                                  low_amp_cover=0.60,
                                  # NEW:
                                  cluster_metric="sad",
                                  cluster_radius=None,
                                  show_debug=True):
        """Run a one-shot scan, summarize the results, and optionally plot them.

        Args:
            name: Label used in plot titles and summary output.
            data: Input signal to scan.
            threshold: Minimum Pearson correlation required for a candidate.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            rel_err: Maximum allowed relative RMS error after template fitting.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches.
            cluster_radius: Maximum distance between nearby matches before clustering.
            show_debug: Whether to show debugging plots.

        Returns:
            Accepted match indices and their corresponding correlation scores.

        References:
            NumPy Developers. `numpy.lib.stride_tricks.sliding_window_view`.
            Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
        """

        # Scanning data with template for matches
        idx, sco = self._scan_matches(
            data,
            threshold,
            amp_max,
            rel_err,
            pos_shift=pos_shift,
            sad_thresh=sad_thresh,
            min_std=min_std,
            enforce_low_amp=enforce_low_amp,
            low_amp_quantile=low_amp_quantile,
            low_amp_cover=low_amp_cover,
            # ↓↓↓
            cluster_metric=cluster_metric,
            cluster_radius=cluster_radius,
            # ↑↑↑
            show_debug=show_debug,
            debug_verbose=False
        )

        if show_debug:
            self._debug_plots(signal_data = data, mark_data = np.array([]), horiz_axis_line_inds = [],
                            vert_axis_line_inds = idx, title = f"{name}  final matches (n={len(idx)})",
                            label = 0, plot_option = "matches")

        if sco.size:
            print(f"✅ {len(idx)} matches | mean r = {sco.mean():.4f}")
        else:
            print("No matches found")

        return idx, sco

    """
    Do `passes` of scan→update on a single dataset, then finalize and plot.

    Notes
    -----
    - Template is adapted after each pass from ALL accepted windows.
    - Clustering is applied during scanning to reduce near-duplicates.
    """
    def _run_dataset(self, data, name,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=True,
                    low_amp_quantile=0.10,
                    low_amp_cover=0.60,
                    # :
                    cluster_metric="sad",
                    cluster_radius=None,
                    show_debug=True, debug_verbose=False):
        """Run iterative template adaptation on a dataset and return final matches.

        Args:
            data: Input signal to process.
            name: Label used in plots and summary output.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            thr: Minimum Pearson correlation required for a candidate.
            rel_err: Maximum allowed relative RMS error after template fitting.
            passes: Number of scan-update passes to run before the final scan.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches.
            cluster_radius: Maximum distance between nearby matches before clustering.
            show_debug: Whether to show debugging plots during processing.
            debug_verbose: Whether to print verbose progress and scan diagnostics.

        Returns:
            Final accepted match indices and their corresponding correlation scores.

        References:
            NumPy Developers. `numpy.lib.stride_tricks.sliding_window_view`.
        """

        for p in range(passes):
            if debug_verbose:
                print(f"   pass {p+1}/{passes}")
            idx, _ = self._scan_matches(
                data, thr, amp_max, rel_err,
                pos_shift=pos_shift,
                sad_thresh=sad_thresh,
                min_std=min_std,
                enforce_low_amp=enforce_low_amp,
                low_amp_quantile=low_amp_quantile,
                low_amp_cover=low_amp_cover,
                # ↓↓↓
                cluster_metric=cluster_metric,
                cluster_radius=cluster_radius,
                # ↑↑↑
                show_debug=show_debug,
                debug_verbose=debug_verbose
            )
            self._update_template(data, idx, show_debug=show_debug)
        
        # Final pass of the refined template
        final_marked_indices, final_marked_indices_scores = self._final_template_match_and_plot(
            name, data, thr, amp_max,
            pos_shift, sad_thresh, min_std, rel_err,
            enforce_low_amp, low_amp_quantile, low_amp_cover,
            # ↓↓↓
            cluster_metric=cluster_metric,
            cluster_radius=cluster_radius,
            # ↑↑↑
            show_debug=show_debug
        )
        self.found_indices_scores = final_marked_indices_scores
        return final_marked_indices

    
    def cold_start_run_dataset(self, data_dict: dict,
                    amp_max: float, thr: float, rel_err: float, passes: int,
                    pos_shift: float, sad_thresh: float, min_std: float,
                    enforce_low_amp: bool=True,
                    low_amp_quantile: float=0.10,
                    low_amp_cover: float=0.60,
                    # :
                    cluster_metric: str="sad",
                    cluster_radius: int=None,
                    show_debug: bool=False, debug_verbose: bool=False):
        """Run the dataset pipeline by _resetting the template to the starting piecewise linear function before each dataset.

        Args:
            data_dict: Mapping of dataset names to input signal arrays.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            thr: Minimum Pearson correlation required for a candidate.
            rel_err: Maximum allowed relative RMS error after template fitting.
            passes: Number of scan-update passes to run before the final scan.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches.
            cluster_radius: Maximum distance between nearby matches before clustering.
            show_debug: Whether to show debugging plots during processing.
            debug_verbose: Whether to print verbose progress and scan diagnostics.

        Returns:
            Dictionary of dataset names mapped to final match results.

        References:
            NumPy Developers. `numpy.ndarray`.
        """
        # _Reset template shape in case new class instance isn't used for a new run.
        self._reset()

        return_dict = {}
        for dataKey in data_dict:
            return_dict[dataKey] = self._run_dataset(data_dict[dataKey], dataKey,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=enforce_low_amp,
                    low_amp_quantile=low_amp_quantile,
                    low_amp_cover=low_amp_cover,
                    # :
                    cluster_metric=cluster_metric,
                    cluster_radius=cluster_radius,
                    show_debug=show_debug, debug_verbose=show_debug
            )

            self._reset()

        return return_dict

    def warm_start_run_dataset(self, data_dict: dict, name: str,
                    amp_max: float, thr: float, rel_err: float, passes: int,
                    pos_shift: float, sad_thresh: float, min_std: float,
                    enforce_low_amp: bool=True,
                    low_amp_quantile: float=0.10,
                    low_amp_cover: float=0.60,
                    # :
                    cluster_metric: str="sad",
                    cluster_radius: int=None,
                    show_debug: bool=True, debug_verbose: bool=False):
        """Warm-start the template on one dataset, then evaluate all datasets.

        Args:
            data_dict: Mapping of dataset names to input signal arrays.
            name: Dataset name to use for template warm-starting, or `None` to choose one automatically.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            thr: Minimum Pearson correlation required for a candidate.
            rel_err: Maximum allowed relative RMS error after template fitting.
            passes: Number of scan-update passes to run during warm-starting.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches.
            cluster_radius: Maximum distance between nearby matches before clustering.
            show_debug: Whether to show debugging plots during processing.
            debug_verbose: Whether to print verbose progress and scan diagnostics.

        Returns:
            Dictionary of dataset names mapped to final match results.

        References:
            Python Software Foundation. `random.choice`.
        """
        # _Reset template shape in case new class instance isn't used for a new run.
        self._reset()

        datasetToTrain = name
        # Choose a random dataset to adapt the template on
        if datasetToTrain is None:
            datasetToTrain = random.choice(list(data_dict.keys()))

        # Update the template on the selected dataset
        self._run_dataset(data_dict[datasetToTrain], datasetToTrain,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=enforce_low_amp,
                    low_amp_quantile=low_amp_quantile,
                    low_amp_cover=low_amp_cover,
                    # :
                    cluster_metric=cluster_metric,
                    cluster_radius=cluster_radius,
                    show_debug=show_debug, debug_verbose=show_debug
            )

        # Don't perform any more updates
        passes = 0
        return_dict = {}

        for dataKey in data_dict:
            return_dict[dataKey] = self._run_dataset(data_dict[dataKey], dataKey,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=enforce_low_amp,
                    low_amp_quantile=low_amp_quantile,
                    low_amp_cover=low_amp_cover,
                    # :
                    cluster_metric=cluster_metric,
                    cluster_radius=cluster_radius,
                    show_debug=show_debug, debug_verbose=show_debug
            )

        return return_dict

    
    def all_data_run_dataset(self, data_dict: dict,
                    amp_max: float, thr: float, rel_err: float, passes: int,
                    pos_shift: float, sad_thresh: float, min_std: float,
                    enforce_low_amp: bool=True,
                    low_amp_quantile: float=0.10,
                    low_amp_cover: float=0.60,
                    # :
                    cluster_metric: str="sad",
                    cluster_radius: int=None,
                    show_debug: bool=True, debug_verbose: bool=False):
        """Adapt the template across all datasets, then rerun matching on all datasets.

        Args:
            data_dict: Mapping of dataset names to input signal arrays.
            amp_max: Maximum allowed absolute amplitude within a candidate window.
            thr: Minimum Pearson correlation required for a candidate.
            rel_err: Maximum allowed relative RMS error after template fitting.
            passes: Number of scan-update passes to run for each dataset during adaptation.
            pos_shift: Constant added to the SAD trace for thresholding and plotting.
            sad_thresh: Minimum SAD score required to keep a candidate.
            min_std: Minimum standard deviation required for a candidate window.
            enforce_low_amp: Whether to require low-amplitude coverage in each window.
            low_amp_quantile: Quantile used to define low-amplitude samples.
            low_amp_cover: Minimum fraction of samples that must be below the low-amplitude threshold.
            cluster_metric: Metric used to merge nearby matches.
            cluster_radius: Maximum distance between nearby matches before clustering.
            show_debug: Whether to show debugging plots during processing.
            debug_verbose: Whether to print verbose progress and scan diagnostics.

        Returns:
            Dictionary of dataset names mapped to final match results.

        References:
            NumPy Developers. `numpy.ndarray`.
        """
        # _Reset template shape in case new class instance isn't used for a new run.
        self._reset()

        # Update the template on all the datasets first before a final parse
        for dataKey in data_dict:
           self._run_dataset(data_dict[dataKey], dataKey,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=enforce_low_amp,
                    low_amp_quantile=low_amp_quantile,
                    low_amp_cover=low_amp_cover,
                    # :
                    cluster_metric=cluster_metric,
                    cluster_radius=cluster_radius,
                    show_debug=show_debug, debug_verbose=show_debug
            )

        # Don't perform any more updates, use the template updated across all datasets
        passes = 0
        return_dict = {}

        for dataKey in data_dict:
            return_dict[dataKey] = self._run_dataset(data_dict[dataKey], dataKey,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=enforce_low_amp,
                    low_amp_quantile=low_amp_quantile,
                    low_amp_cover=low_amp_cover,
                    # :
                    cluster_metric=cluster_metric,
                    cluster_radius=cluster_radius,
                    show_debug=show_debug, debug_verbose=show_debug
            )

        return return_dict

    def get_template_shape(self):
        """Return the current template shape as a NumPy array.

        Args:
            None.

        Returns:
            The current template shape.
        """
        return self.template

    def get_final_overlap_signal(self):
        """Return the final scores for each found index.

        Args:
            None.

        Returns:
            The final scores for found best overlap indices.
        """
        return self.found_indices_scores
