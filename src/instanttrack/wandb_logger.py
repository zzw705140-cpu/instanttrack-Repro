"""独立的 W&B 日志工具；训练与评估的接入由调用方负责。

认证沿用 RoboForge 的方式：由 W&B SDK 使用运行机器已有的凭据。
本模块不读取、保存或打印令牌，也不显式调用登录接口。
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from socket import gethostname
from typing import Any
from uuid import uuid4


def _recursive_flatten_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """把嵌套指标转换为 ``train/loss`` 形式的键名。"""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            nested = _recursive_flatten_dict(value)
            flattened.update({f"{key}/{name}": item for name, item in nested.items()})
        else:
            flattened[str(key)] = value
    return flattened


class WandBLogger:
    """封装实验初始化、配置同步、指标记录与结束。

    ``debug=True`` 使用 SDK 的 disabled 模式，不向线上上传。
    ``run_id`` 与 ``resume`` 可供后续训练、评估共用实验身份；本模块
    不决定身份文件的保存位置，也不扫描或上传本地目录。
    """

    def __init__(
        self,
        *,
        project: str = "instanttrack",
        entity: str | None = None,
        group: str | None = None,
        description: str = "instanttrack",
        variant: Mapping[str, Any] | None = None,
        wandb_output_dir: str | Path | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        resume: str | None = None,
        debug: bool = False,
    ) -> None:
        if not project.strip():
            raise ValueError("W&B project must not be empty.")
        if resume not in (None, "allow", "must", "never", "auto"):
            raise ValueError("Unsupported W&B resume mode.")
        if resume is not None and run_id is None:
            raise ValueError("Provide run_id when resuming a W&B run.")

        # 延迟导入：尚未接入 W&B 的训练代码不需要安装其 SDK。
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("W&B logging requires the 'wandb' package.") from exc

        output_dir = None
        if wandb_output_dir is not None:
            directory = Path(wandb_output_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            output_dir = str(directory)

        # 只同步调用方明确提供的配置，不收集环境变量或命令行凭据。
        config = dict(variant) if variant is not None else {}
        config.setdefault("hostname", gethostname())
        self.experiment_id = run_id if run_id is not None else uuid4().hex
        self.run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            tags=[description] if description else None,
            config=config,
            dir=output_dir,
            id=self.experiment_id,
            **({"name": run_name if run_name is not None else self.experiment_id}
               if run_name is not None or resume is None else {}),
            mode="disabled" if debug else "online",
            resume=resume,
        )

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        """上传一组指标；调用方控制记录频率与步数。"""
        self.run.log(_recursive_flatten_dict(data), step=step)

    def finish(self) -> None:
        """等待剩余日志写入并结束实验。"""
        self.run.finish()


def make_wandb_logger(**kwargs: Any) -> WandBLogger:
    """创建日志工具，参数与 ``WandBLogger`` 一致。"""
    return WandBLogger(**kwargs)


def create_training_logger(args: Any) -> WandBLogger | None:
    """根据训练参数创建实验；身份单独保存，供后续评估使用。"""
    if not (args.wandb or args.wandb_debug):
        return None
    directory = Path(args.log_dir)
    identity_path = directory / "wandb_run.json"
    previous = None
    if not args.wandb_debug:
        if args.resume:
            source = Path(args.resume).parent / "wandb_run.json"
            if source.exists():
                previous = json.loads(source.read_text())
                if previous["project"] != args.wandb_project:
                    raise ValueError("W&B project differs from the resumed run.")
                if args.wandb_entity is not None and previous["entity"] != args.wandb_entity:
                    raise ValueError("W&B entity differs from the resumed run.")
        if identity_path.exists():
            if previous is None or json.loads(identity_path.read_text()) != previous:
                raise FileExistsError("Log directory already belongs to a W&B run; resume it or choose another directory.")

    logger = make_wandb_logger(
        project=args.wandb_project,
        entity=previous["entity"] if previous else args.wandb_entity,
        variant=vars(args),
        wandb_output_dir=directory,
        run_name=args.wandb_name or directory.resolve().name,
        run_id=previous["id"] if previous else None,
        resume="must" if previous else None,
        debug=args.wandb_debug,
    )
    try:
        # SDK 提交步数自动增长；显式训练迭代允许恢复和后续评估使用同一横轴。
        logger.run.define_metric("iteration")
        logger.run.define_metric("train/*", step_metric="iteration")
        if not args.wandb_debug:
            identity = {"id": logger.run.id, "project": logger.run.project, "entity": logger.run.entity}
            temporary = identity_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(identity, indent=2))
            temporary.replace(identity_path)
    except BaseException:
        logger.finish()
        raise
    return logger


def create_evaluation_logger(run_dir: str | Path, *, debug: bool = False, reps: int = 1) -> WandBLogger:
    """关联训练实验；不覆盖训练配置，也不自动创建新的线上实验。"""
    directory = Path(run_dir)
    identity = None
    if not debug:
        identity = json.loads((directory / "wandb_run.json").read_text())
    logger = make_wandb_logger(
        project=identity["project"] if identity else "instanttrack",
        entity=identity["entity"] if identity else None,
        run_id=identity["id"] if identity else None,
        resume="must" if identity else None,
        description="",  # 保留训练实验已有的标签和显示名。
        run_name=None,
        wandb_output_dir=directory,
        debug=debug,
    )
    try:
        prefix = "eval" if reps == 1 else f"eval_r{reps}"
        logger.run.define_metric(f"{prefix}/iteration")
        logger.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/iteration")
    except BaseException:
        logger.finish()
        raise
    return logger


def log_evaluation_metrics(logger: WandBLogger, result: Mapping[str, Any], *,
                           iteration: int, train_wall_h: float, n_clips: int, reps: int = 1) -> None:
    """完成率为未失败比例；多副本时使用平均失败数，与 CSV 的口径一致。"""
    if n_clips <= 0 or reps <= 0:
        raise ValueError("n_clips and reps must be positive.")
    n_failed = float(result["n_failed"])
    if not 0 <= n_failed <= n_clips:
        raise ValueError("Evaluation failure count is outside the clip count.")
    prefix = "eval" if reps == 1 else f"eval_r{reps}"
    logger.log({prefix: {
        "iteration": iteration,
        "train_wall_h": train_wall_h,
        "norm_return": float(result["norm_return"]),
        "norm_return_framew": float(result["norm_return_framew"]),
        "n_failed": n_failed,
        "completion_rate": 1.0 - n_failed / n_clips,
        "n_clips": n_clips,
        "reps": reps,
    }})


__all__ = ["WandBLogger", "make_wandb_logger", "create_training_logger",
           "create_evaluation_logger", "log_evaluation_metrics"]
