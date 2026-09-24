"""Đánh giá attack detection ở cấp session (binary: có tấn công hay không)."""


class SessionEvaluator:
    """
    Dùng lại ở cả 02_correlation_engine.py (đánh giá ngay sau khi build) và
    04_evaluate_pipeline.py (đánh giá lại từ file đã lưu) để không lặp code
    tính TP/FP/FN/TN + precision/recall/F1.
    """

    @staticmethod
    def binary_metrics(has_pred_attack, has_gt_attack):
        """
        has_pred_attack / has_gt_attack: pandas Series bool cùng index.
        Trả về dict TP/FP/FN/TN + precision/recall/F1.
        """
        tp = int((has_pred_attack & has_gt_attack).sum())
        fp = int((has_pred_attack & ~has_gt_attack).sum())
        fn = int((~has_pred_attack & has_gt_attack).sum())
        tn = int((~has_pred_attack & ~has_gt_attack).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)

    @classmethod
    def from_dataframe(cls, df, pred_col="has_pred_attack", gt_col="has_gt_attack"):
        return cls.binary_metrics(df[pred_col], df[gt_col])

    @staticmethod
    def print_metrics(label, metrics):
        print(f"\n[{label}]")
        print(f"  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "
              f"FN={metrics['fn']:,}  TN={metrics['tn']:,}")
        print(f"  Precision={metrics['precision']:.3f}  "
              f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
