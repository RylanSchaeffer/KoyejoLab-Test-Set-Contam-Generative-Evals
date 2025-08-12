import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import wandb

import src.analyze
import src.plot


# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "0r7lchfg",  # Qwen 2.5 3B.
]

run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)
run_configs_df["eval/neg_log_mean_token_accuracy"] = -np.log(
    run_configs_df["eval/mean_token_accuracy"],
)
run_configs_df["Num. Train Epochs"] = run_configs_df["num_train_epochs"]
run_configs_df = run_configs_df[run_configs_df["seed"] == 0]


plt.close()
g = sns.scatterplot(
    data=run_configs_df, x="Num. Train Epochs", y="eval/mean_token_accuracy"
)
g.set(
    ylim=(0.75, 1.0),
    xscale="log",
    xlabel="Num. Train Epochs",
    ylabel="Token Accuracy",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=mean_token_accuracy_x=num_train_epochs",
)
# plt.show()

plt.close()
g = sns.scatterplot(data=run_configs_df, x="Num. Train Epochs", y="eval/loss")
g.set(
    xscale="log",
    yscale="log",
    xlabel="Num. Train Epochs",
    ylabel="Test Loss",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=test_loss_x=num_train_epochs",
)
plt.show()


plt.close()
g = sns.scatterplot(
    data=run_configs_df, x="Num. Train Epochs", y="eval/neg_log_mean_token_accuracy"
)
g.set(
    xscale="log",
    yscale="log",
    xlabel="Num. Train Epochs",
    ylabel=r"$-\log ( \text{Token Accuracy} )$",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg_log_mean_token_accuracy_x=num_train_epochs",
)
# plt.show()

for temperature in ["0.0", "0.316", "1.0"]:
    plt.close()
    g = sns.scatterplot(
        data=run_configs_df,
        x="Num. Train Epochs",
        y=f"lm_eval_after_temp={temperature}/math_verify_none",
    )
    g.set(
        xscale="log",
        xlabel="Num. Train Epochs",
        ylabel="Exact Match",
        ylim=(-0.05, 1.05),
    )
    avg_math_verify_none_before = run_configs_df[
        f"lm_eval_before_temp={temperature}/math_verify_none"
    ].mean()
    plt.axhline(avg_math_verify_none_before, color="k", linestyle="--")
    plt.text(1.05, avg_math_verify_none_before - 0.1, "Starting Checkpoint")
    plt.title(f"Temperature: {temperature}")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y=em_after_x=num_train_epochs_temp={temperature}",
    )
    # plt.show()

    plt.close()
    plt.figure(figsize=(8, 6))
    g = sns.scatterplot(
        data=run_configs_df,
        x=f"lm_eval_before_temp={temperature}/math_verify_none",
        y=f"lm_eval_after_temp={temperature}/math_verify_none",
        hue="Num. Train Epochs",
        hue_norm=matplotlib.colors.LogNorm(),
        palette="copper",
    )
    g.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Exact Match (Before)",
        ylabel="Exact Match (After)",
    )
    plt.title(f"Temperature: {temperature}")
    plt.plot([0.01, 1.0], [0.01, 1.0], linestyle="--", color="black")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title="Num.\nTrain\nEpochs")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y=em_after_x=em_before_hue=num_train_epochs_temp={temperature}",
    )
    # plt.show()

print("Finished 00_gsm8k_platinum")
