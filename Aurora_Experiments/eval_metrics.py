import numpy as np
import glob as _glob
import torch
import torch_harmonics as th

OMEGA = 7.292e-5      # Earth angular velocity, rad/s
A_EARTH = 6.371e6     # Earth radius, m
G = 9.80665           # Earth gravity, m/s^2
R_D = 287.05          # specific gas constant for dry air, J/(kg K)


def lat_weighted_rmse(pred, y, vars, lat, log_postfix):
    error = (pred - y) ** 2

    # latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat/w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    return loss_dict


def bias(pred, y, vars, log_postfix):
    # Creates bias map for plotting
    b = pred - y

    bias_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            bias_dict[f"bias_{var}_{log_postfix}"] = b[:, i]
    return bias_dict


def lat_weighted_rmsb(pred, y, vars, lat, log_postfix):
    rmsb_dict = {}
    error = torch.mean(pred - y, dim=0) ** 2  # [V, H, W]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat/w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # [H, 1]

    with torch.no_grad():
        for i, var in enumerate(vars):
            rmsb_dict[f"rmsb_{var}_{log_postfix}"] = torch.sqrt(torch.mean(error[i] * w_lat))
    return rmsb_dict

def lat_weighted_signed_bias(pred, y, vars, lat, log_postfix):
    err = pred - y                                          # [B, V, H, W]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(
        dtype=err.dtype, device=err.device)                 # [1, H, 1]

    out = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            out[f"sbias_{var}_{log_postfix}"] = torch.mean(
                torch.mean(err[:, i] * w_lat, dim=(-2, -1)))
    return out


def acc(pred, y, clim, vars, lat, log_postfix):
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)

    clim = clim.to(pred.device)
    pred_a = pred - clim.unsqueeze(0)
    y_a = y - clim.unsqueeze(0)

    acc_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            # Skip vars without a climatology slice (NaN sentinel from caller)
            if torch.isnan(clim[i]).any():
                continue
            num = torch.sum(w_lat * pred_a[:, i] * y_a[:, i])
            den = torch.sqrt(torch.sum(w_lat * pred_a[:, i] ** 2) * torch.sum(w_lat * y_a[:, i] ** 2))
            acc_dict[f"acc_{var}_{log_postfix}"] = num / den
    return acc_dict

def power_spectrum(field, lat):
    field = field - field.mean(dim=-1, keepdim=True)  # remove zonal mean per row

    W = field.shape[-1]
    fft = torch.fft.rfft(field, dim=-1)               # [B, H, W//2+1]
    psd = (fft.abs() ** 2) / (W ** 2)                  # Parseval normalization
    if W % 2 == 0:
        psd[..., 1:-1] = psd[..., 1:-1] * 2
    else:
        psd[..., 1:] = psd[..., 1:] * 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.sum()
    w_lat = torch.from_numpy(w_lat).to(dtype=psd.dtype, device=psd.device)
    w_lat = w_lat.unsqueeze(0).unsqueeze(-1)           # [1, H, 1]

    psd = (psd * w_lat).sum(dim=1).mean(dim=0)        # lat-weighted mean over H, then batch mean -> [W//2+1]

    wavenumbers = torch.arange(1, psd.shape[-1])       # drop k=0 (DC)
    return wavenumbers, psd[1:]

def calculate_power_spectrums(pred, y, vars, lat, log_postfix):
    psd_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            wavenumbers, psd = power_spectrum(pred[:, i], lat)
            _, psd_gt = power_spectrum(y[:, i], lat)
            psd_dict[f"psd_preds_{var}_{log_postfix}"] = psd
            psd_dict[f"psd_truth_{var}_{log_postfix}"] = psd_gt
            psd_dict[f"wavenumbers_{var}_{log_postfix}"] = wavenumbers
            psd_dict[f"spectral_ratio_{var}_{log_postfix}"] = psd/psd_gt
    return psd_dict


_SHT_CACHE = {}


def _get_sht(nlat, nlon, device, dtype):
    key = (nlat, nlon, device.type, str(dtype))
    sht = _SHT_CACHE.get(key)
    if sht is None:
        sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device=device, dtype=dtype)
        _SHT_CACHE[key] = sht
    return sht


def _sht_degrees(nlat):
    """Total-wavenumber axis l = 1..l_max (l=0 is the global mean, dropped); l_max = nlat-1."""
    return torch.arange(1, nlat)


def _sht_power(fields, lat):
    """Isotropic SH power spectrum of a batch of fields.
    fields: [N, H, W] real (physical units). Returns [N, L] power at l = 1..l_max."""
    lat = np.asarray(lat)
    if lat[0] > lat[-1]:                          # ERA5 is N->S; torch-harmonics grid is S->N
        fields = torch.flip(fields, dims=(-2,))
    H, W = fields.shape[-2], fields.shape[-1]
    sht = _get_sht(H, W, fields.device, fields.dtype)
    c = sht(fields)                               # [N, lmax, mmax] complex, lmax = H
    p = c[..., 0].abs() ** 2 + 2.0 * c[..., 1:].abs().pow(2).sum(-1)   # [N, lmax]
    p = p / (4.0 * np.pi)                          # -> area-weighted variance units
    return p[..., 1:]                             # [N, L], drop l=0 (global mean)


def sh_power_spectrum(field, lat):
    """Spherical-harmonic power spectrum of a [B, H, W] field (batch-averaged).
    Mirrors `power_spectrum`; returns (degrees, power) indexed by total wavenumber
    l = 1..l_max."""
    degrees = _sht_degrees(field.shape[-2])
    power = _sht_power(field, lat).mean(dim=0)    # [L]
    return degrees, power


def sh_wavelength_km(degrees):
    """Physical horizontal wavelength (km) for total wavenumber l: 2*pi*R/sqrt(l(l+1))."""
    l = torch.as_tensor(degrees, dtype=torch.float64)
    return (2.0 * np.pi * A_EARTH / torch.sqrt(l * (l + 1.0)) / 1000.0)


def calculate_sh_power_spectrums(pred, y, vars, lat, log_postfix):
    B, V = pred.shape[0], pred.shape[1]
    psd_dict = {}
    with torch.no_grad():
        degrees = _sht_degrees(pred.shape[-2])
        stack = torch.cat([pred.reshape(B * V, *pred.shape[-2:]),
                           y.reshape(B * V, *y.shape[-2:])], dim=0)
        S = _sht_power(stack, lat)                            # [2*B*V, L]
        psd = S[: B * V].reshape(B, V, -1).mean(dim=0)        # [V, L]
        psd_gt = S[B * V:].reshape(B, V, -1).mean(dim=0)      # [V, L]

        psd_dict[f"sh_degrees_{log_postfix}"] = degrees
        psd_dict[f"sh_wavelength_km_{log_postfix}"] = sh_wavelength_km(degrees)
        for i, var in enumerate(vars):
            psd_dict[f"sh_psd_preds_{var}_{log_postfix}"] = psd[i]
            psd_dict[f"sh_psd_truth_{var}_{log_postfix}"] = psd_gt[i]
            psd_dict[f"sh_spectral_ratio_{var}_{log_postfix}"] = psd[i] / psd_gt[i]
    return psd_dict

def spectral_div(pred, y, vars, lat, log_postfix):
    spectral_div = {}

    def _kl(psd, psd_gt):
        psd = psd / psd.sum()
        psd_gt = psd_gt / psd_gt.sum()
        div = torch.sum(psd_gt * torch.log(torch.clamp(psd_gt / psd, min=1e-9)))
        div_back = torch.sum(psd * torch.log(torch.clamp(psd / psd_gt, min=1e-9)))
        return div, div_back

    B, V = pred.shape[0], pred.shape[1]
    with torch.no_grad():
        stack = torch.cat([pred.reshape(B * V, *pred.shape[-2:]),
                           y.reshape(B * V, *y.shape[-2:])], dim=0)
        S = _sht_power(stack, lat)                            # [2*B*V, L]
        sh_psd_all = S[: B * V].reshape(B, V, -1).mean(dim=0)     # [V, L]
        sh_psd_gt_all = S[B * V:].reshape(B, V, -1).mean(dim=0)   # [V, L]

        for i, var in enumerate(vars):
            _, psd = power_spectrum(pred[:, i], lat)
            _, psd_gt = power_spectrum(y[:, i], lat)
            div, div_back = _kl(psd, psd_gt)
            spectral_div[f"spec_div_{var}_{log_postfix}"] = div
            spectral_div[f"spec_div_back_{var}_{log_postfix}"] = div_back

            sh_div, sh_div_back = _kl(sh_psd_all[i], sh_psd_gt_all[i])
            spectral_div[f"sh_spec_div_{var}_{log_postfix}"] = sh_div
            spectral_div[f"sh_spec_div_back_{var}_{log_postfix}"] = sh_div_back
    return spectral_div

def spectral_res(pred, y, vars, lat, log_postfix):
    """RMS residual between the normalized pred and truth spectra, computed with BOTH the
    zonal FFT (`spec_res*`) and the spherical-harmonic transform (`sh_spec_res*`). The SH
    version is the headline; the zonal one is kept for the appendix similarity check."""
    spectral_res = {}

    def _res(psd, psd_gt):
        psd = psd / psd.sum()
        psd_gt = psd_gt / psd_gt.sum()
        return torch.sqrt(torch.mean((psd - psd_gt) ** 2))

    B, V = pred.shape[0], pred.shape[1]
    with torch.no_grad():
        stack = torch.cat([pred.reshape(B * V, *pred.shape[-2:]),
                           y.reshape(B * V, *y.shape[-2:])], dim=0)
        S = _sht_power(stack, lat)                            # [2*B*V, L]
        sh_psd = S[: B * V].reshape(B, V, -1).mean(dim=0)     # [V, L]
        sh_psd_gt = S[B * V:].reshape(B, V, -1).mean(dim=0)

        for i, var in enumerate(vars):
            _, psd = power_spectrum(pred[:, i], lat)
            _, psd_gt = power_spectrum(y[:, i], lat)
            spectral_res[f"spec_res_{var}_{log_postfix}"] = _res(psd, psd_gt)
            spectral_res[f"sh_spec_res_{var}_{log_postfix}"] = _res(sh_psd[i], sh_psd_gt[i])
    return spectral_res

def spec_res_log(psd_pred, psd_truth, eps=1e-30, l_max=None):
    p = np.asarray(psd_pred, dtype=np.float64)
    t = np.asarray(psd_truth, dtype=np.float64)
    if l_max is not None:
        p, t = p[:l_max], t[:l_max]
    return float(np.sqrt(np.mean((np.log(p + eps) - np.log(t + eps)) ** 2)))


def spec_div_w1(psd_pred, psd_truth):
    p = np.asarray(psd_pred, dtype=np.float64)
    t = np.asarray(psd_truth, dtype=np.float64)
    p = p / p.sum()
    t = t / t.sum()
    return float(np.sum(np.abs(np.cumsum(t) - np.cumsum(p))))

def effective_resolution_wavelength(ratio, wavelength_km, threshold=0.5):
    ratio = np.asarray(ratio)
    below = ratio < threshold
    if not below.any():
        return float(wavelength_km[-1])          # resolves down to the grid
    # index where the final below-threshold run begins (robust to large-scale wobble)
    idx = len(below) - 1
    while idx > 0 and below[idx - 1]:
        idx -= 1
    return float(wavelength_km[idx])


def predictable_scale_wavelength(dke, bg, wavelength_km, frac=0.5):
    dke = np.asarray(dke)
    bg = np.asarray(bg)
    saturated = dke >= frac * 2.0 * bg
    if not saturated.any():
        return float(wavelength_km[-1])
    idx = len(saturated) - 1
    while idx > 0 and saturated[idx - 1]:
        idx -= 1
    return float(wavelength_km[idx])


def rqe(pred, y, vars, log_postfix, quantiles=(0.9, 0.99, 0.999, 0.9999)):
    rqe_dict = {}
    q_tensor = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    with torch.no_grad():
        B, V = pred.shape[0], pred.shape[1]
        p_flat = pred.reshape(B, V, -1)
        y_flat = y.reshape(B, V, -1)
        p_q = torch.quantile(p_flat, q_tensor, dim=-1)  # [Q, B, V]
        y_q = torch.quantile(y_flat, q_tensor, dim=-1)
        rel = (p_q - y_q) / y_q                         # [Q, B, V]
        rqe_per_sample = rel.sum(dim=0)                 # [B, V]
        p_q_mean = p_q.mean(dim=1).cpu()                # [Q, V]
        y_q_mean = y_q.mean(dim=1).cpu()
        for i, var in enumerate(vars):
            rqe_dict[f"rqe_{var}_{log_postfix}"] = rqe_per_sample[:, i].mean()
            rqe_dict[f"q_pred_{var}_{log_postfix}"] = p_q_mean[:, i]
            rqe_dict[f"q_true_{var}_{log_postfix}"] = y_q_mean[:, i]
        rqe_dict[f"quantiles_{log_postfix}"] = q_tensor.cpu()
    return rqe_dict

def _spherical_grads(field, lat):
    lat = np.asarray(lat)
    W = field.shape[-1]
    phi = torch.from_numpy(np.deg2rad(lat)).to(dtype=field.dtype, device=field.device)
    cosphi = torch.cos(phi).clamp(min=1e-6).view(-1, 1)   # [H,1], floored so poles don't blow up
    dphi = float(np.deg2rad(lat[1] - lat[0]))             # signed scalar spacing
    dlam = 2.0 * np.pi / W                                # uniform longitude spacing

    # d/dlambda: periodic central difference along the W axis
    df_dlam = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / (2.0 * dlam)

    # d/dphi: central difference along the H axis, one-sided at the edges
    df_dphi = torch.empty_like(field)
    df_dphi[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dphi)
    df_dphi[..., 0, :]    = (field[..., 1, :]  - field[..., 0, :])  / dphi
    df_dphi[..., -1, :]   = (field[..., -1, :] - field[..., -2, :]) / dphi

    dfdx = df_dlam / (A_EARTH * cosphi)
    dfdy = df_dphi / A_EARTH
    return dfdx, dfdy


def wind_balance(pred, y, vars, lat, log_postfix):
    lat = np.asarray(lat)

    # cos-lat weight, zeroed outside the extra-tropical band, normalised to sum 1 over H
    band = np.abs(lat) >= 20.0
    w = np.cos(np.deg2rad(lat)) * band
    w = w / w.sum()
    w_lat = torch.from_numpy(w).to(dtype=pred.dtype, device=pred.device)  # [H]

    phi = torch.from_numpy(np.deg2rad(lat)).to(dtype=pred.dtype, device=pred.device).view(-1, 1)
    f = 2.0 * OMEGA * torch.sin(phi)                                      # [H,1]
    f = torch.where(f.abs() < 1e-10, torch.full_like(f, 1e-10), f)        # floor away from 0

    def _wmean(mag):                                     # [B,H,W] -> [B] area-weighted mean
        return (mag.mean(dim=-1) * w_lat).sum(dim=-1)

    def _levels_for(prefix):                             # {level: var index} for present fields
        out = {}
        for i, v in enumerate(vars):
            if v.startswith(prefix):
                tail = v[len(prefix):]
                if tail.isdigit():
                    out[int(tail)] = i
        return out

    z_idx = _levels_for("geopotential_")
    u_idx = _levels_for("u_component_of_wind_")
    v_idx = _levels_for("v_component_of_wind_")
    levels = sorted(set(z_idx) & set(u_idx) & set(v_idx))

    result = {f"wbal_levels_{log_postfix}": levels}
    with torch.no_grad():
        for src, tag in ((pred, "pred"), (y, "truth")):
            for L in levels:
                z = src[:, z_idx[L]]                     # [B,H,W]
                u = src[:, u_idx[L]]
                v = src[:, v_idx[L]]
                dfdx, dfdy = _spherical_grads(z, lat)
                ug = -dfdy / f
                vg = dfdx / f
                # Geostrophic wind
                mag_vg = torch.sqrt(ug ** 2 + vg ** 2)
                # Ageostrophic wind
                mag_vag = torch.sqrt((u - ug) ** 2 + (v - vg) ** 2)
                vg_mean = _wmean(mag_vg).mean()         # cos-lat extra-tropical mean |Vg| (m/s)
                vag_mean = _wmean(mag_vag).mean()        # ... and |Vag| (m/s)
                result[f"wbal_vg_{tag}_{L}_{log_postfix}"] = vg_mean
                result[f"wbal_vag_{tag}_{L}_{log_postfix}"] = vag_mean
                result[f"wbal_ageo_geo_{tag}_{L}_{log_postfix}"] = vag_mean / vg_mean
    return result

def _levels_for(vars, prefix):
    """{level: var index} for variables named '<prefix><level>' (e.g. 'temperature_850')."""
    out = {}
    for i, v in enumerate(vars):
        if v.startswith(prefix):
            tail = v[len(prefix):]
            if tail.isdigit():
                out[int(tail)] = i
    return out


def _cell_area(lat, W):
    """Grid-cell area A_i = R^2 * |d(sin phi)| * d(lambda), numpy [H] in m^2.
    Central differences for interior rows, one-sided at the poles (Sha et al. Eq A2);
    summed over the grid this recovers 4*pi*R^2 to ~0.02%."""
    lat_arr = np.asarray(lat)
    s = np.sin(np.deg2rad(lat_arr))
    dsin = np.empty_like(s)
    dsin[1:-1] = (s[2:] - s[:-2]) / 2.0
    dsin[0] = s[1] - s[0]
    dsin[-1] = s[-1] - s[-2]
    return (A_EARTH ** 2) * np.abs(dsin) * (2.0 * np.pi / W)


def global_dry_air_mass(pred, y, vars, lat, log_postfix):

    q_idx = _levels_for(vars, "specific_humidity_")
    levels = sorted(set(q_idx))                          # ascending hPa -> ascending pressure

    result = {f"dryair_levels_{log_postfix}": levels}
    if len(levels) < 2:                                  # need >= 2 levels to integrate in p
        return result

    with torch.no_grad():
        # Pressure axis (hPa -> Pa), ascending to match `levels`.
        p = torch.tensor([L * 100.0 for L in levels], dtype=pred.dtype, device=pred.device)

        # Area element [H, 1] in m^2 per cell, broadcasts over longitude.
        area = torch.from_numpy(_cell_area(lat, pred.shape[-1])).to(
            dtype=pred.dtype, device=pred.device).unsqueeze(-1)

        md = {}
        for src, tag in ((pred, "pred"), (y, "truth")):
            q = torch.stack([src[:, q_idx[L]] for L in levels], dim=1)   # [B, L, H, W]
            colmass = torch.trapezoid(1.0 - q, p, dim=1) / G            # [B, H, W], kg m^-2
            md[tag] = (colmass * area).sum(dim=(-2, -1))                 # [B], kg

        result[f"dryair_Md_pred_{log_postfix}"] = md["pred"].mean().cpu()
        result[f"dryair_Md_truth_{log_postfix}"] = md["truth"].mean().cpu()
        result[f"dryair_Md_err_{log_postfix}"] = (md["pred"] - md["truth"]).mean().cpu()
    return result


def q_bias_levels(pred, y, vars, lat, log_postfix):
    q_idx = _levels_for(vars, "specific_humidity_")
    levels = sorted(set(q_idx))

    result = {f"qbias_levels_{log_postfix}": levels,
              f"qbias_lat_{log_postfix}": np.asarray(lat)}
    if len(levels) < 2:
        return result

    with torch.no_grad():
        p = torch.tensor([L * 100.0 for L in levels], dtype=pred.dtype, device=pred.device)
        # Trapezoid half-interval weights: integral f dp ~= sum_k w_k f_k.
        w_p = torch.empty_like(p)
        w_p[1:-1] = (p[2:] - p[:-2]) / 2.0
        w_p[0] = (p[1] - p[0]) / 2.0
        w_p[-1] = (p[-1] - p[-2]) / 2.0

        area = torch.from_numpy(_cell_area(lat, pred.shape[-1])).to(
            dtype=pred.dtype, device=pred.device).unsqueeze(-1)          # [H, 1]
        total_area = area.sum() * pred.shape[-1]                          # cells share area along lon

        for k, L in enumerate(levels):
            dq = pred[:, q_idx[L]] - y[:, q_idx[L]]                       # [B, H, W]
            wsum = (dq * area).sum(dim=(-2, -1))                          # [B], kg/kg * m^2
            result[f"qbias_mean_{L}_{log_postfix}"] = (wsum / total_area).mean().cpu()
            result[f"qbias_truth_{L}_{log_postfix}"] = (
                (y[:, q_idx[L]] * area).sum(dim=(-2, -1)) / total_area).mean().cpu()
            result[f"qbias_md_contrib_{L}_{log_postfix}"] = (-(w_p[k] / G) * wsum).mean().cpu()
            result[f"qbias_zonal_{L}_{log_postfix}"] = dq.mean(dim=(0, -1)).cpu()  # [H]
    return result


def divergence_vorticity(pred, y, vars, lat, log_postfix):
    lat = np.asarray(lat)
    band = np.abs(lat) >= 20.0
    w = np.cos(np.deg2rad(lat)) * band
    w = w / w.sum()

    u_idx = _levels_for(vars, "u_component_of_wind_")
    v_idx = _levels_for(vars, "v_component_of_wind_")
    levels = sorted(set(u_idx) & set(v_idx))

    result = {f"divvort_levels_{log_postfix}": levels}
    with torch.no_grad():
        w_lat = torch.from_numpy(w).to(dtype=pred.dtype, device=pred.device)  # [H]

        def _wmean(mag):                                  # [B,H,W] -> scalar
            return ((mag.mean(dim=-1) * w_lat).sum(dim=-1)).mean()

        for src, tag in ((pred, "pred"), (y, "truth")):
            for L in levels:
                u = src[:, u_idx[L]]
                v = src[:, v_idx[L]]
                dudx, dudy = _spherical_grads(u, lat)
                dvdx, dvdy = _spherical_grads(v, lat)
                div_mean = _wmean(torch.abs(dudx + dvdy))   # s^-1
                vort_mean = _wmean(torch.abs(dvdx - dudy))  # s^-1
                result[f"divvort_div_{tag}_{L}_{log_postfix}"] = div_mean.cpu()
                result[f"divvort_vort_{tag}_{L}_{log_postfix}"] = vort_mean.cpu()
                result[f"divvort_ratio_{tag}_{L}_{log_postfix}"] = (div_mean / vort_mean).cpu()
    return result


def difference_kinetic_energy(pred, y, vars, lat, log_postfix):
    u_idx = _levels_for(vars, "u_component_of_wind_")
    v_idx = _levels_for(vars, "v_component_of_wind_")
    levels = sorted(set(u_idx) & set(v_idx))

    result = {f"dke_levels_{log_postfix}": levels}
    if not levels:
        return result

    with torch.no_grad():
        wavenumbers = None
        for L in levels:
            up, vp = pred[:, u_idx[L]], pred[:, v_idx[L]]
            ut, vt = y[:, u_idx[L]], y[:, v_idx[L]]

            wn, psd_du = power_spectrum(up - ut, lat)
            _, psd_dv = power_spectrum(vp - vt, lat)
            _, psd_ut = power_spectrum(ut, lat)
            _, psd_vt = power_spectrum(vt, lat)
            _, psd_up = power_spectrum(up, lat)
            _, psd_vp = power_spectrum(vp, lat)

            if wavenumbers is None:
                wavenumbers = wn

            result[f"dke_dke_{L}_{log_postfix}"] = (0.5 * (psd_du + psd_dv)).cpu()
            result[f"dke_bg_{L}_{log_postfix}"] = (0.5 * (psd_ut + psd_vt)).cpu()
            result[f"dke_fc_{L}_{log_postfix}"] = (0.5 * (psd_up + psd_vp)).cpu()

        result[f"dke_wavenumbers_{log_postfix}"] = wavenumbers.cpu()

        # --- SH DKE spectra (headline), all levels x {du,dv,ut,vt,up,vp} in one transform ---
        B = pred.shape[0]
        sh_fields = []
        for L in levels:
            up, vp = pred[:, u_idx[L]], pred[:, v_idx[L]]
            ut, vt = y[:, u_idx[L]], y[:, v_idx[L]]
            sh_fields += [up - ut, vp - vt, ut, vt, up, vp]      # each [B, H, W]
        Ssh = _sht_power(torch.cat(sh_fields, dim=0), lat)       # [6*nlev*B, Ldeg]
        Ssh = Ssh.reshape(len(levels), 6, B, -1).mean(dim=2)     # [nlev, 6, Ldeg] (batch mean)
        for j, L in enumerate(levels):
            du, dv, ut_, vt_, up_, vp_ = Ssh[j]
            result[f"sh_dke_dke_{L}_{log_postfix}"] = (0.5 * (du + dv)).cpu()
            result[f"sh_dke_bg_{L}_{log_postfix}"] = (0.5 * (ut_ + vt_)).cpu()
            result[f"sh_dke_fc_{L}_{log_postfix}"] = (0.5 * (up_ + vp_)).cpu()
        result[f"sh_dke_degrees_{log_postfix}"] = _sht_degrees(pred.shape[-2]).cpu()
    return result


def hypsometric_residual(pred, y, vars, lat, log_postfix, virtual=True):
    z_idx = _levels_for(vars, "geopotential_")
    t_idx = _levels_for(vars, "temperature_")
    q_idx = _levels_for(vars, "specific_humidity_")
    levels = sorted(set(z_idx) & set(t_idx))             # ascending pressure (hPa)
    pairs = list(zip(levels[:-1], levels[1:]))           # (p_up, p_dn), p_up < p_dn

    lat = np.asarray(lat)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()

    def _tv(src, L):
        """Layer temperature at level L: virtual T if requested and q is present, else dry T."""
        t = src[:, t_idx[L]]
        if virtual and L in q_idx:
            return t * (1.0 + 0.6078 * src[:, q_idx[L]])
        return t

    result = {f"hyps_pairs_{log_postfix}": [f"{dn}-{up}" for up, dn in pairs]}
    with torch.no_grad():
        w_lat = torch.from_numpy(w).to(dtype=pred.dtype, device=pred.device)  # [H]

        def _wrms(x):                                     # [B,H,W] -> scalar lat-weighted RMS
            return torch.sqrt(((x ** 2).mean(dim=-1) * w_lat).sum(dim=-1)).mean()

        for src, tag in ((pred, "pred"), (y, "truth")):
            for up, dn in pairs:
                thick = src[:, z_idx[up]] - src[:, z_idx[dn]]                 # m^2/s^2, > 0
                t_bar = 0.5 * (_tv(src, up) + _tv(src, dn))
                hyps = R_D * t_bar * np.log(dn / up)
                pair_key = f"{dn}-{up}"
                result[f"hyps_rms_{tag}_{pair_key}_{log_postfix}"] = _wrms(thick - hyps).cpu()
                result[f"hyps_thick_{tag}_{pair_key}_{log_postfix}"] = _wrms(thick).cpu()
    return result


REGIONS_LAPSE = {"tropics": (-30.0, 30.0), "nh_mid": (30.0, 60.0),
                 "sh_mid": (-60.0, -30.0)}


def _weighted_w1_1d(vp, wp, vt, wt):
    vp = np.asarray(vp, dtype=np.float64); wp = np.asarray(wp, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.float64); wt = np.asarray(wt, dtype=np.float64)
    vals = np.concatenate([vp, vt])
    order = np.argsort(vals)
    vals = vals[order]
    cw_p = np.concatenate([wp / wp.sum(), np.zeros_like(wt)])[order]
    cw_t = np.concatenate([np.zeros_like(wp), wt / wt.sum()])[order]
    cdf_p = np.cumsum(cw_p)
    cdf_t = np.cumsum(cw_t)
    dv = np.diff(vals)
    return float(np.sum(np.abs(cdf_p - cdf_t)[:-1] * dv))


def lapse_rate_wasserstein(pred, y, vars, lat, log_postfix):
    z_idx = _levels_for(vars, "geopotential_")
    t_idx = _levels_for(vars, "temperature_")
    result = {f"lapse_regions_{log_postfix}": list(REGIONS_LAPSE)}
    need = {500, 850}
    if not (need <= set(z_idx) and need <= set(t_idx)):
        return result

    lat = np.asarray(lat)
    coslat = np.cos(np.deg2rad(lat))
    with torch.no_grad():
        def gamma(src):
            dT = src[:, t_idx[500]] - src[:, t_idx[850]]
            dPhi = src[:, z_idx[500]] - src[:, z_idx[850]]
            return (-G * dT / dPhi * 1000.0)                 # [B, H, W], K/km

        g_p = gamma(pred).detach().cpu().numpy()
        g_t = gamma(y).detach().cpu().numpy()
        B = g_p.shape[0]
        w2d = np.broadcast_to(coslat[:, None], g_p.shape[-2:])   # [H, W]

        regional = []
        for name, (lo, hi) in REGIONS_LAPSE.items():
            mask = (lat >= lo) & (lat <= hi)                    # [H]
            m2d = np.broadcast_to(mask[:, None], g_p.shape[-2:])
            wsel = (w2d * m2d).reshape(-1)
            keep = wsel > 0
            wcell = wsel[keep]
            if wcell.size == 0:
                w1 = float("nan")
            else:
                gp = g_p.reshape(B, -1)[:, keep].reshape(-1)
                gt = g_t.reshape(B, -1)[:, keep].reshape(-1)
                wtile = np.tile(wcell, B)                       # weights repeat per batch
                w1 = _weighted_w1_1d(gp, wtile, gt, wtile)
            result[f"lapse_w1_{name}_{log_postfix}"] = torch.tensor(
                w1, dtype=pred.dtype, device=pred.device).cpu()
            regional.append(w1)
        result[f"lapse_w1_mean_{log_postfix}"] = torch.tensor(
            float(np.mean(regional)), dtype=pred.dtype, device=pred.device).cpu()
    return result


def negative_humidity(pred, y, vars, lat, log_postfix):
    q_idx = _levels_for(vars, "specific_humidity_")
    levels = sorted(set(q_idx))

    result = {f"negq_levels_{log_postfix}": levels}
    if len(levels) < 2:
        return result

    with torch.no_grad():
        p = torch.tensor([L * 100.0 for L in levels], dtype=pred.dtype, device=pred.device)
        area = torch.from_numpy(_cell_area(lat, pred.shape[-1])).to(
            dtype=pred.dtype, device=pred.device).unsqueeze(-1)               # [H, 1]

        for src, tag in ((pred, "pred"), (y, "truth")):
            q = torch.stack([src[:, q_idx[L]] for L in levels], dim=1)        # [B, L, H, W]
            result[f"negq_frac_{tag}_{log_postfix}"] = (q < 0).float().mean().cpu()
            q_neg = torch.clamp(q, max=0.0)                                   # only the q<0 part
            col = torch.trapezoid(q_neg, p, dim=1) / G                        # [B, H, W], kg m^-2
            result[f"negq_mass_{tag}_{log_postfix}"] = (col * area).sum(dim=(-2, -1)).mean().cpu()
    return result


def compute_all_metrics(pred, y, clim, vars, lat, lead_time):
    metrics = {
        "RMSE": lat_weighted_rmse(pred, y, vars, lat, log_postfix=str(lead_time)),
        "bias": bias(pred, y, vars, log_postfix=str(lead_time)),
        "signed_bias": lat_weighted_signed_bias(pred, y, vars, lat,
                                                log_postfix=str(lead_time)),
        "lat_weighted_rmsb": lat_weighted_rmsb(pred, y, vars, lat, log_postfix=str(lead_time)),
        "power spectrum": calculate_power_spectrums(pred, y, vars, lat, log_postfix=str(lead_time)),
        "sh power spectrum": calculate_sh_power_spectrums(pred, y, vars, lat, log_postfix=str(lead_time)),
        "RQE": rqe(pred, y, vars, log_postfix=str(lead_time)),
        "spec_div": spectral_div(pred, y, vars, lat, log_postfix=str(lead_time)),
        "spec_res": spectral_res(pred, y, vars, lat, log_postfix=str(lead_time)),
        "wind_balance": wind_balance(pred, y, vars, lat, log_postfix=str(lead_time)),
        "dry_air_mass": global_dry_air_mass(pred, y, vars, lat, log_postfix=str(lead_time)),
        "q_bias": q_bias_levels(pred, y, vars, lat, log_postfix=str(lead_time)),
        "div_vort": divergence_vorticity(pred, y, vars, lat, log_postfix=str(lead_time)),
        "dke": difference_kinetic_energy(pred, y, vars, lat, log_postfix=str(lead_time)),
        "hypsometric": hypsometric_residual(pred, y, vars, lat, log_postfix=str(lead_time)),
        "lapse_rate": lapse_rate_wasserstein(pred, y, vars, lat, log_postfix=str(lead_time)),
        "neg_humidity": negative_humidity(pred, y, vars, lat, log_postfix=str(lead_time)),
    }
    if clim is not None:
        metrics["ACC"] = acc(pred, y, clim, vars, lat, log_postfix=str(lead_time))
    return metrics

