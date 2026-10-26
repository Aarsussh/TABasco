"""Generate the polished two-column report PDF from saved TABasco result CSVs.

The companion report.tex contains the equivalent submission source; this builder
keeps the PDF reproducible on machines without a TeX installation.
"""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = REPORT / "report.pdf"
SUMMARY = ROOT / "results/sensitivity/heldout_constraint/evaluation_summary.csv"
PLOTS = ROOT / "results/sensitivity/plots/heldout"
AUTHOR = "Aarussh Shrivastava (B24CM1001) | IIT Jodhpur"
GITHUB = "https://github.com/Aarsussh/TABasco"


def rows_by_policy():
    with SUMMARY.open(newline="", encoding="utf-8-sig") as stream:
        return {row["policy"]: row for row in csv.DictReader(stream)}


def val(data, policy, field, digits=2):
    return f"{float(data[policy][field]):.{digits}f}"


class TwoColumnReport(BaseDocTemplate):
    def __init__(self, path):
        left, right, top, bottom, gap = 34, 34, 32, 30, 16
        super().__init__(str(path), pagesize=letter, leftMargin=left, rightMargin=right,
                         topMargin=top, bottomMargin=bottom, title="TABasco: CSL3090 Assignment 2",
                         author=AUTHOR, subject="Task-aware browser tab optimization with DQN")
        width = (letter[0] - left - right - gap) / 2
        height = letter[1] - top - bottom
        frames = [Frame(left, bottom, width, height, id="left", leftPadding=1,
                        rightPadding=1, topPadding=0, bottomPadding=0),
                  Frame(left + width + gap, bottom, width, height, id="right", leftPadding=1,
                        rightPadding=1, topPadding=0, bottomPadding=0)]
        self.addPageTemplates([PageTemplate(id="two-column", frames=frames, onPage=self.footer)])

    @staticmethod
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(letter[0] / 2, 16, f"TABasco | CSL3090 | {doc.page}")
        canvas.restoreState()


def main():
    data = rows_by_policy()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ProjectTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=14, leading=16, alignment=TA_CENTER, spaceAfter=3))
    styles.add(ParagraphStyle(name="Identity", parent=styles["Normal"], fontSize=7.4, leading=9,
                              alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=7.5,
                              leading=9.0, alignment=TA_JUSTIFY, spaceAfter=4))
    styles.add(ParagraphStyle(name="SectionSmall", parent=styles["Heading2"], fontSize=9.5,
                              leading=11, spaceBefore=4, spaceAfter=3,
                              textColor=colors.HexColor("#17324d")))
    styles.add(ParagraphStyle(name="CaptionSmall", parent=styles["BodyText"], fontSize=6.8,
                              leading=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Reference", parent=styles["BodyText"], fontSize=9,
                              leading=12, spaceAfter=8))
    story = []

    def head(text):
        story.append(Paragraph(text, styles["SectionSmall"]))

    def para(text):
        story.append(Paragraph(text, styles["BodySmall"]))

    story += [Paragraph("TABasco: Task-Aware Browser Tab Optimization using Reinforcement Learning",
                        styles["ProjectTitle"]),
              Paragraph(f"{AUTHOR}<br/>CSL3090 Artificial Intelligence, Assignment 2 | October 2026<br/>{GITHUB}",
                        styles["Identity"])]
    head("Abstract")
    para("TABasco is an original synthetic browser-tab management MDP trained with a Deep Q-Network (DQN) [1]. In an initial soft-budget evaluation, DQN earned higher composite return while violating the memory budget more often. Increasing the penalty reduced but did not remove violations, so a constraint-aware transition variant was trained. On the previously examined 60-session evaluation block, it maintained the simulator's active-memory invariant. DQN had higher composite return than suspension baselines, although those baselines retained a larger fraction of useful revisits. These results describe the simulator and reward, not a real browser.")

    head("1. MDP and simulator")
    para("For N=12 slots, state s has 8N+5=101 float32 values. Each slot records occupancy, active status, normalized memory, recency, access frequency, current-task match, estimated revisit probability, and unsaved-work risk. Five global values encode active memory/budget, task, remaining horizon, elapsed fraction, and occupancy. Tab sizes range from 0.45 to 1.35 synthetic units. Tasks switch; tabs arrive, age, expire, and may be revisited. Revisit probability changes with age, frequency, and task affinity.")
    para("There are 3N+1=37 actions: KEEP, SUSPEND, or CLOSE for one slot, plus WAIT. KEEP retains an active tab or restores a suspended one; active KEEP is a charged no-op. SUSPEND requires an active tab, CLOSE requires occupancy, and WAIT is always valid. Empty-slot actions and redundant suspension are invalid. Each decision advances one step; episodes end at a fixed 120-step horizon, not when all tabs close.")
    para("Only active tabs count against B=8; suspension frees the full tab size and suspended metadata costs zero active memory. Suspended revisits restore the tab and charge restoration. Closed pages retain 30-step history; recovery charges restoration and lost-work costs. Expiration charges risk-weighted loss and retains recovery history. All future revisit and arrival events are sampled from the evolving session state.")
    head("2. Reward and constraint handling")
    para("Reward is the negative sum of positive costs: r = -[c<sub>m</sub> max(0,M-B) + C<sub>restore</sub> + 1.2 C<sub>lost</sub> + 0.45 C<sub>disrupt</sub> + C<sub>action</sub> + C<sub>deferred</sub>]. The selected c<sub>m</sub>=1; restoration multiplier is 0.8, ordinary action/WAIT cost is 0.035, invalid action cost is 0.12, and constrained-event deferral cost is 0.25. Per-step reward components sum to the returned reward. There is no tab-count bonus or terminal reward.")
    para("The penalty sweep (coefficients 1, 3, 5; two fresh training seeds each) left 11.18%, 9.89%, and 8.59% of validation steps over budget. Stronger penalties alone were insufficient. In the selected constraint-aware variant, initial overflow tabs start suspended; infeasible restores are masked; suspended revisits that cannot fit stay suspended; closed recovery remains in history if it cannot fit; and over-budget arrivals are declined, not queued. Every transition checks active memory after all automatic events. The guarantee is specific to this simulator's active-memory model.")
    story.append(FrameBreak())
    head("3. DQN and baselines")
    para("The DQN is a 101-128-128-37 ReLU MLP with Adam (0.001), discount 0.97, batch size 64, replay capacity 50,000, Huber loss, gradient clipping at 10, and target updates every 500 learner updates. Epsilon decays from 1.0 to 0.05 over 20,000 decisions. The same observation-derived action mask is applied in exploration, greedy selection, and replay bootstrap. Runs used CPU, 250 episodes of up to 120 steps, and fresh network seeds 3090 and 4090.")
    para("LRU suspends the eligible active tab with the oldest last visit; LFU uses the lowest access count then oldest; Random samples uniformly from this same eligible set. All WAIT when no active tab is eligible. They are evaluation baselines, not training labels. Policies share the same initial scenario and event seed per episode; DQN is greedy at evaluation.")
    head("4. Design evolution and model selection")
    para("The first version explored empty-slot actions without masking, could end when all tabs closed, and did not account consistently for no-op and expiry costs. Corrected runs use one validity mask, a fixed horizon, and explicit reward-component checks. The soft-budget evaluation still exceeded capacity because arrivals and automatic returns could increase memory after the agent decision; raising the penalty from 1 to 5 only reduced validation violation frequency from 11.18% to 8.59%.")
    para("Constraint-aware transitions were introduced to meet the explicit capacity requirement while making infeasible returns visible as deferrals. A later reward-weight experiment did not support replacing the selected model: disruption-weight 0.35 changed paired return by +2.43 for training seed 3090 and -1.47 for seed 4090; revisit success declined. The two-seed average is descriptive and uncertain.")
    head("Policy update")
    para("The finite-horizon objective maximizes expected discounted return, E[sum gamma^t r_t], with gamma=0.97. Replay stores state, action, reward, next state, terminal flag, and next-state action mask. DQN minimizes Huber loss against target r + gamma(1-done) max valid-action Q-target(next state). A target network supplies bootstrap values; epsilon-greedy exploration samples valid actions only [1,2].")
    head("Transition sequence")
    para("After the management action, time advances, the task may switch, and revisit probabilities update. Bernoulli draws determine candidates; at most one open-tab revisit is selected, preferring the highest current probability. Closed-history recovery may then occur, followed by at most one new arrival and stochastic age-based expiration. Reward uses final active memory. In soft mode, automatic restoration or arrival can exceed the budget after the decision; a tab-by-tab policy cannot preempt that event. Constraint-aware mode instead defers or declines infeasible events and checks the post-event memory bound.")
    story.append(PageBreak())
    head("5. Measured results")
    para("The selected checkpoint evaluation used seeds 91000-91059. This block was examined during earlier project analysis, so it is a previously examined evaluation, not a newly untouched test. Table 1 reports scenario mean and SD.")

    table_data = [["Policy", "Return +/- SD", "Mem.", "Useful", "Viol."]]
    for policy, label in [("DQN_seed3090", "DQN 3090"), ("DQN_seed4090", "DQN 4090"),
                          ("LRU", "LRU"), ("LFU", "LFU"), ("Random", "Random")]:
        table_data.append([label,
                           f"{val(data, policy, 'return_mean')} +/- {val(data, policy, 'return_std')}",
                           val(data, policy, "mean_active_memory_mean"),
                           val(data, policy, "useful_revisit_success_rate_mean", 3),
                           val(data, policy, "violation_steps_mean")])
    table = Table(table_data, colWidths=[57, 91, 37, 42, 34], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("FONTSIZE", (0, 0), (-1, -1), 6.2),
                               ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                                [colors.white, colors.HexColor("#eef3f7")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 2),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story += [table, Paragraph("Table 1. Common-seed policy comparison; useful is task-matching revisit success.", styles["CaptionSmall"])]
    para("Every policy had zero violation steps and zero invalid actions under constraint-aware rules. DQN useful-revisit success was 0.824/0.849, below LRU/LFU/Random at 0.886/0.886/0.887. DQN restoration costs were 5.69/4.37 and disruption costs 3.67/2.07; baselines paid restoration 16.32-16.35 and disruption 36.17-36.30. DQN lost-work costs were higher (6.73/4.33 versus about 1.69), and deferred-event costs were 1.41/1.34. Return therefore reflects a reward-weighted tradeoff, not universal superiority.")
    return_plot = PLOTS / "return_comparison.png"
    if return_plot.exists():
        story.extend([Image(str(return_plot), width=3.05 * inch, height=1.48 * inch),
                      Paragraph("Figure 1. Mean return and 95% scenario confidence intervals.", styles["CaptionSmall"])])
    story.append(FrameBreak())
    head("Reward cost profile")
    cost_data = [["Policy", "Restore", "Lost", "Disrupt", "Defer"]]
    for policy, label in [("DQN_seed3090", "DQN 3090"), ("DQN_seed4090", "DQN 4090"),
                          ("LRU", "LRU"), ("LFU", "LFU"), ("Random", "Random")]:
        cost_data.append([label, val(data, policy, "restoration_recovery_cost_mean"),
                          val(data, policy, "lost_work_cost_mean"),
                          val(data, policy, "task_disruption_cost_mean"),
                          val(data, policy, "deferred_event_cost_mean")])
    cost_table = Table(cost_data, colWidths=[55, 49, 42, 49, 42], repeatRows=1)
    cost_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0, 0), (-1, -1), 6.1),
                                    ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey),
                                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                                     [colors.white, colors.HexColor("#eef3f7")]),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.extend([cost_table, Paragraph("Mean episode cost components; memory cost is zero for every policy under the hard constraint; action cost is 4.20 each.", styles["CaptionSmall"])])
    memory_plot = PLOTS / "memory_comparison.png"
    if memory_plot.exists():
        story.extend([Image(str(memory_plot), width=3.05 * inch, height=1.48 * inch),
                      Paragraph("Figure 2. Mean active memory and 95% intervals; budget is 8.", styles["CaptionSmall"])])
    head("6. Interpretation and limits")
    para("The later reward-weight validation did not justify replacing the selected model. Lowering the disruption coefficient to 0.35 changed paired canonical return by +2.43 for training seed 3090 (95% scenario CI +0.26 to +4.59) and -1.47 for seed 4090 (-3.43 to +0.49), mean +0.48 over only two training seeds. Useful revisit fell from 0.820/0.834 to 0.797/0.802. The effect reversed by seed and remained uncertain; current checkpoints were retained. The prior evaluation block did not inform that validation choice.")
    para("Capacity safety is enforced by the environment, not learned by DQN. The workload and memory units are synthetic; scenario intervals do not measure training-seed uncertainty. More seeds, calibrated workloads, and memory measurements are needed before deployment claims.")

    story.append(PageBreak())
    story.append(Paragraph("References", styles["SectionSmall"]))
    story.append(Paragraph("[1] V. Mnih et al., “Human-level control through deep reinforcement learning,” <i>Nature</i>, vol. 518, pp. 529-533, 2015. https://doi.org/10.1038/nature14236.", styles["Reference"]))
    story.append(Paragraph("[2] R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, 2nd ed. MIT Press, 2018.", styles["Reference"]))
    doc = TwoColumnReport(OUT)
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
