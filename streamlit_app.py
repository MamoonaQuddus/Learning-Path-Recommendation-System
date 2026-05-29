import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Learning Path Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Learning Path Recommendation System")
st.markdown(
    """
This application recommends personalized learning courses for interns
using **Collaborative Filtering** and **Matrix Factorization (Truncated SVD)**.
"""
)


# ============================================================
# RECOMMENDER SYSTEM CLASS
# ============================================================

class LearningPathRecommender:

    def __init__(
        self,
        interactions_df,
        courses_df,
        n_components=20,
        random_state=42,
    ):

        self.interactions_df = interactions_df
        self.courses_df = courses_df
        self.n_components = n_components
        self.random_state = random_state

        self.intern_index_ = None
        self.course_index_ = None

        self.svd = None
        self.intern_factors_ = None
        self.course_factors_ = None

    def _build_interaction_matrix(self):

        pivot_df = self.interactions_df.pivot_table(
            index="intern_id",
            columns="course_id",
            values="rating",
            aggfunc="mean"
        )

        self.intern_index_ = pivot_df.index
        self.course_index_ = pivot_df.columns

        interaction_matrix = pivot_df.fillna(0.0).to_numpy()

        return interaction_matrix

    def fit(self):

        interaction_matrix = self._build_interaction_matrix()

        n_components = min(
            self.n_components,
            min(interaction_matrix.shape) - 1
        )

        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.random_state
        )

        self.intern_factors_ = self.svd.fit_transform(
            interaction_matrix
        )

        self.course_factors_ = self.svd.components_.T

    def _predict_scores_for_all_courses(self):

        score_matrix = np.dot(
            self.intern_factors_,
            self.course_factors_.T
        )

        scaler = MinMaxScaler()

        score_matrix_scaled = scaler.fit_transform(score_matrix)

        return score_matrix_scaled

    def recommend_for_intern(
        self,
        intern_id,
        top_n=5,
        filter_by_difficulty=None,
        preferred_category=None,
    ):

        if intern_id not in self.intern_index_:
            return pd.DataFrame()

        intern_pos = self.intern_index_.get_loc(intern_id)

        score_matrix = self._predict_scores_for_all_courses()

        intern_scores = score_matrix[intern_pos]

        seen_courses = self.interactions_df.loc[
            self.interactions_df["intern_id"] == intern_id,
            "course_id"
        ].unique()

        recommendations = []

        for col_idx, course_id in enumerate(self.course_index_):

            if course_id in seen_courses:
                continue

            score = intern_scores[col_idx]

            course_row = self.courses_df[
                self.courses_df["course_id"] == course_id
            ]

            if course_row.empty:
                continue

            course_row = course_row.iloc[0]

            rec = {
                "course_id": course_id,
                "title": course_row.get("title", ""),
                "category": course_row.get("category", ""),
                "difficulty": course_row.get("difficulty", ""),
                "duration_hours": course_row.get("duration_hours", np.nan),
                "predicted_score": round(float(score), 3),
            }

            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)

        if rec_df.empty:
            return rec_df

        if filter_by_difficulty and filter_by_difficulty != "All":
            rec_df = rec_df[
                rec_df["difficulty"].str.lower()
                == filter_by_difficulty.lower()
            ]

        if preferred_category and preferred_category != "All":
            rec_df = rec_df[
                rec_df["category"].str.lower()
                == preferred_category.lower()
            ]

        rec_df = rec_df.sort_values(
            by="predicted_score",
            ascending=False
        )

        return rec_df.head(top_n).reset_index(drop=True)

    def build_learning_path(
        self,
        intern_id,
        top_n=5,
        strategy="difficulty_ascending",
    ):

        rec_df = self.recommend_for_intern(
            intern_id=intern_id,
            top_n=top_n * 3
        )

        if rec_df.empty:
            return rec_df

        if strategy == "difficulty_ascending":

            difficulty_order = {
                "beginner": 0,
                "intermediate": 1,
                "advanced": 2
            }

            rec_df["difficulty_rank"] = rec_df[
                "difficulty"
            ].apply(
                lambda x: difficulty_order.get(
                    str(x).lower(),
                    1
                )
            )

            rec_df = rec_df.sort_values(
                by=["difficulty_rank", "predicted_score"],
                ascending=[True, False]
            )

            rec_df = rec_df.drop(columns=["difficulty_rank"])

        elif strategy == "shortest_first":

            rec_df = rec_df.sort_values(
                by=["duration_hours", "predicted_score"],
                ascending=[True, False]
            )

        return rec_df.head(top_n).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📂 Upload Dataset Files")

courses_file = st.sidebar.file_uploader(
    "Upload course_metadata.csv",
    type=["csv"]
)

ratings_file = st.sidebar.file_uploader(
    "Upload intern_ratings.csv",
    type=["csv"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if courses_file and ratings_file:

    courses_df = pd.read_csv(courses_file)
    ratings_df = pd.read_csv(ratings_file)

    st.success("Datasets uploaded successfully!")

    # Display datasets
    with st.expander("📘 Course Metadata"):
        st.dataframe(courses_df)

    with st.expander("⭐ Intern Ratings"):
        st.dataframe(ratings_df)

    # Initialize model
    recommender = LearningPathRecommender(
        interactions_df=ratings_df,
        courses_df=courses_df,
        n_components=20,
        random_state=42
    )

    recommender.fit()

    st.sidebar.header("⚙️ Recommendation Settings")

    intern_ids = sorted(ratings_df["intern_id"].unique())

    selected_intern = st.sidebar.selectbox(
        "Select Intern ID",
        intern_ids
    )

    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    difficulty_filter = st.sidebar.selectbox(
        "Filter by Difficulty",
        ["All", "Beginner", "Intermediate", "Advanced"]
    )

    categories = ["All"] + sorted(
        courses_df["category"].dropna().unique().tolist()
    )

    category_filter = st.sidebar.selectbox(
        "Filter by Category",
        categories
    )

    strategy = st.sidebar.selectbox(
        "Learning Path Strategy",
        ["difficulty_ascending", "shortest_first"]
    )

    # Recommendations
    st.subheader("📚 Recommended Courses")

    recommendations = recommender.recommend_for_intern(
        intern_id=selected_intern,
        top_n=top_n,
        filter_by_difficulty=difficulty_filter,
        preferred_category=category_filter,
    )

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(recommendations)

    # Learning path
    st.subheader("🛤️ Personalized Learning Path")

    learning_path = recommender.build_learning_path(
        intern_id=selected_intern,
        top_n=top_n,
        strategy=strategy,
    )

    if learning_path.empty:
        st.warning("No learning path available.")
    else:
        st.dataframe(learning_path)

else:
    st.info(
        "Please upload both CSV files to start the recommendation system."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, Collaborative Filtering, and Truncated SVD"
)
