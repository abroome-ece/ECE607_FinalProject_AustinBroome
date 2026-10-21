import matplotlib.pyplot as plt


def plot_individual_height_trajectories(
    growth,
    n_children=50
):
    children = (
        growth["anon_id"]
        .drop_duplicates()
        .sample(
            n=min(
                n_children,
                growth["anon_id"].nunique()
            ),
            random_state=42
        )
    )

    plt.figure(figsize=(10, 6))

    for child in children:
        data = growth[
            growth["anon_id"] == child
        ]

        plt.plot(
            data["AgeBabyDays"],
            data["Jit_Heightcm"],
            alpha=0.4
        )

    plt.xlabel("Age (days)")
    plt.ylabel("Height (cm)")
    plt.title(
        f"Height Growth Trajectories "
        f"({len(children)} Random Children)"
    )

    plt.tight_layout()
    plt.show()


def plot_individual_weight_trajectories(
    growth,
    n_children=50
):
    children = (
        growth["anon_id"]
        .drop_duplicates()
        .sample(
            n=min(
                n_children,
                growth["anon_id"].nunique()
            ),
            random_state=42
        )
    )

    plt.figure(figsize=(10, 6))

    for child in children:
        data = growth[
            growth["anon_id"] == child
        ]

        plt.plot(
            data["AgeBabyDays"],
            data["jit_weightkg"],
            alpha=0.4
        )

    plt.xlabel("Age (days)")
    plt.ylabel("Weight (kg)")
    plt.title(
        f"Weight Growth Trajectories "
        f"({len(children)} Random Children)"
    )

    plt.tight_layout()
    plt.show()


def plot_height_by_age(growth):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        growth["AgeBabyDays"],
        growth["Jit_Heightcm"],
        alpha=0.05,
        s=5
    )

    plt.xlabel("Age (days)")
    plt.ylabel("Height (cm)")
    plt.title("Height Measurements by Age")

    plt.tight_layout()
    plt.show()


def plot_weight_by_age(growth):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        growth["AgeBabyDays"],
        growth["jit_weightkg"],
        alpha=0.05,
        s=5
    )

    plt.xlabel("Age (days)")
    plt.ylabel("Weight (kg)")
    plt.title("Weight Measurements by Age")

    plt.tight_layout()
    plt.show()


def plot_height_by_age_sex(growth):
    plt.figure(figsize=(10, 6))

    for sex in ["Female", "Male"]:
        data = growth[
            growth["Sex"] == sex
        ]

        plt.scatter(
            data["AgeBabyDays"],
            data["Jit_Heightcm"],
            alpha=0.05,
            s=5,
            label=sex
        )

    plt.xlabel("Age (days)")
    plt.ylabel("Height (cm)")
    plt.title("Height Measurements by Age and Sex")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_weight_by_age_sex(growth):
    plt.figure(figsize=(10, 6))

    for sex in ["Female", "Male"]:
        data = growth[
            growth["Sex"] == sex
        ]

        plt.scatter(
            data["AgeBabyDays"],
            data["jit_weightkg"],
            alpha=0.05,
            s=5,
            label=sex
        )

    plt.xlabel("Age (days)")
    plt.ylabel("Weight (kg)")
    plt.title("Weight Measurements by Age and Sex")
    plt.legend()

    plt.tight_layout()
    plt.show()