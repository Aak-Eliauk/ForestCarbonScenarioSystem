import pandas as pd


class EventTable:
    def __init__(self, event_type, records):
        self.event_type = event_type
        self.records = records

    def count(self):
        count = len(self.records)
        return count

    def summary(self):
        if self.records.empty:
            return {
                "event_type": self.event_type,
                "count": 0,
                "year_min": None,
                "year_max": None,
                "severity_mean": None,
            }

        severity_mean = None
        if "Severity" in self.records.columns:
            severity_mean = float(self.records["Severity"].mean())

        year_min = None
        year_max = None
        if "y_event" in self.records.columns:
            year_min = int(self.records["y_event"].min())
            year_max = int(self.records["y_event"].max())

        return {
            "event_type": self.event_type,
            "count": int(len(self.records)),
            "year_min": year_min,
            "year_max": year_max,
            "severity_mean": severity_mean,
        }

    def annual_summary(self):
        if self.records.empty:
            data = pd.DataFrame(columns=["event_type", "year", "count", "severity_mean"])
            return data

        year_map = {}
        for _, row in self.records.iterrows():
            year = int(row["y_event"])
            if year not in year_map:
                year_map[year] = {
                    "count": 0,
                    "severity_total": 0.0,
                    "severity_count": 0,
                }

            year_map[year]["count"] = year_map[year]["count"] + 1
            if "Severity" in self.records.columns:
                severity_value = row.get("Severity", None)
                if severity_value is not None:
                    year_map[year]["severity_total"] = year_map[year]["severity_total"] + float(severity_value)
                    year_map[year]["severity_count"] = year_map[year]["severity_count"] + 1

        rows = []
        for year in sorted(year_map.keys()):
            severity_mean = None
            if year_map[year]["severity_count"] > 0:
                severity_mean = year_map[year]["severity_total"] / year_map[year]["severity_count"]

            rows.append(
                {
                    "event_type": self.event_type,
                    "year": int(year),
                    "count": int(year_map[year]["count"]),
                    "severity_mean": severity_mean,
                }
            )
        data = pd.DataFrame(rows)
        return data


class SimulationBundle:
    def __init__(
        self,
        scenario_name,
        sim_stack,
        summary,
        event_summaries=None,
        yearly_event_df=None,
        training_summary_df=None,
        training_sample_df=None,
        output_files=None,
    ):
        self.scenario_name = scenario_name
        self.sim_stack = sim_stack
        self.summary = summary
        if event_summaries is None:
            event_summaries = []
        self.event_summaries = event_summaries
        self.yearly_event_df = yearly_event_df
        self.training_summary_df = training_summary_df
        self.training_sample_df = training_sample_df
        if output_files is None:
            output_files = {}
        self.output_files = output_files


class ReportBundle:
    def __init__(
        self,
        scenario_name,
        summary_df,
        total_distribution_df,
        metrics,
        yearly_event_df=None,
        training_summary_df=None,
        training_sample_df=None,
        output_files=None,
    ):
        self.scenario_name = scenario_name
        self.summary_df = summary_df
        self.total_distribution_df = total_distribution_df
        self.metrics = metrics
        self.yearly_event_df = yearly_event_df
        self.training_summary_df = training_summary_df
        self.training_sample_df = training_sample_df
        if output_files is None:
            output_files = {}
        self.output_files = output_files
