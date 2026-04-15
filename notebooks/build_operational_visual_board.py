from __future__ import annotations

import pandas as pd
from IPython.display import HTML


def build_operational_visual_board(
    df: pd.DataFrame,
    decision_date: str = "2019-12-30",
) -> HTML:
    """
    Build a two-panel operational visualization for a selected scoring date.

    The board shows separate T+21 and T+28 decision panels, each with:
    - Direction
    - Reliability
    - Stability
    - Action

    Parameters
    ----------
    df : pd.DataFrame
        Scored operational dataframe containing at least:
        Date, Prob_Up_t21, Prob_Down_t21, Prob_Up_t28, Prob_Down_t28,
        Strength_t21, Strength_t28, Alignment_Band, Guardrail, Calibrated_Action
    decision_date : str, default "2019-12-30"
        Date to display as the operational decision point.

    Returns
    -------
    IPython.display.HTML
        Styled HTML object ready to display in a notebook.
    """
    required_cols = [
        "Date",
        "Prob_Up_t21",
        "Prob_Down_t21",
        "Prob_Up_t28",
        "Prob_Down_t28",
        "Strength_t21",
        "Strength_t28",
        "Alignment_Band",
        "Guardrail",
        "Calibrated_Action",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_local = df.copy()
    df_local["Date"] = pd.to_datetime(df_local["Date"])

    target_date = pd.Timestamp(decision_date)
    row = df_local.loc[df_local["Date"] == target_date]

    if row.empty:
        available_min = df_local["Date"].min()
        available_max = df_local["Date"].max()
        raise ValueError(
            f"No row found for {target_date.date()}. "
            f"Available range is {available_min.date()} to {available_max.date()}."
        )

    row = row.iloc[0]

    def color_from_strength(strength: str) -> str:
        strength = str(strength)
        if "STRONG" in strength:
            return "#2e7d32"  # green
        if "MODERATE" in strength:
            return "#f9a825"  # amber
        return "#c62828"      # red

    def color_from_alignment(alignment: str) -> str:
        alignment = str(alignment).upper()
        if alignment == "HIGH":
            return "#2e7d32"
        if alignment == "MEDIUM":
            return "#f9a825"
        return "#c62828"

    def color_from_guardrail(guardrail: object) -> str:
        try:
            return "#2e7d32" if int(guardrail) == 0 else "#c62828"
        except Exception:
            return "#c62828"

    def color_from_action(action: str) -> str:
        action = str(action)
        if "Increase pricing" in action or "Reduce pricing" in action:
            return "#2e7d32"
        if "Prepare" in action:
            return "#f9a825"
        return "#c62828"

    def badge(color: str) -> str:
        return (
            f'<span style="display:inline-block; width:18px; height:18px; '
            f'border-radius:50%; background:{color};"></span>'
        )

    t21_color = color_from_strength(row["Strength_t21"])
    t28_color = color_from_strength(row["Strength_t28"])
    alignment_color = color_from_alignment(row["Alignment_Band"])
    stability_color = color_from_guardrail(row["Guardrail"])
    action_color = color_from_action(row["Calibrated_Action"])

    guardrail_text = "Stable (Guardrail = 0)" if int(row["Guardrail"]) == 0 else "Unstable (Guardrail = 1)"

    html = f"""
    <div style="font-family: Arial, sans-serif; margin: 12px 0;">
      <h2 style="margin-bottom: 6px;">Operational Direction Dashboard</h2>
      <div style="color: #555; margin-bottom: 16px;">
        Decision date: <b>{target_date.date()}</b>
      </div>

      <div style="display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap;">

        <div style="flex: 1; min-width: 420px; border: 1px solid #ddd; border-radius: 10px; padding: 14px;">
          <h3 style="margin-top: 0;">T+21 Horizon</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Control</th>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Status</th>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Operational Reading</th>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Direction</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(t21_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                Prob Up = {row["Prob_Up_t21"]:.3f}, Prob Down = {row["Prob_Down_t21"]:.3f} → {row["Strength_t21"]}
              </td>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Reliability</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(alignment_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                Recent alignment band = {row["Alignment_Band"]}
              </td>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Stability</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(stability_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                {guardrail_text}
              </td>
            </tr>
            <tr>
              <td style="padding:8px;">Action</td>
              <td style="padding:8px;">{badge(action_color)}</td>
              <td style="padding:8px;">
                {row["Calibrated_Action"]}
              </td>
            </tr>
          </table>
        </div>

        <div style="flex: 1; min-width: 420px; border: 1px solid #ddd; border-radius: 10px; padding: 14px;">
          <h3 style="margin-top: 0;">T+28 Horizon</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Control</th>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Status</th>
              <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Operational Reading</th>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Direction</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(t28_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                Prob Up = {row["Prob_Up_t28"]:.3f}, Prob Down = {row["Prob_Down_t28"]:.3f} → {row["Strength_t28"]}
              </td>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Reliability</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(alignment_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                Recent alignment band = {row["Alignment_Band"]}
              </td>
            </tr>
            <tr>
              <td style="padding:8px; border-bottom:1px solid #eee;">Stability</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">{badge(stability_color)}</td>
              <td style="padding:8px; border-bottom:1px solid #eee;">
                {guardrail_text}
              </td>
            </tr>
            <tr>
              <td style="padding:8px;">Action</td>
              <td style="padding:8px;">{badge(action_color)}</td>
              <td style="padding:8px;">
                {row["Calibrated_Action"]}
              </td>
            </tr>
          </table>
        </div>

      </div>
    </div>
    """

    return HTML(html)