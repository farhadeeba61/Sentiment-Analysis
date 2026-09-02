import streamlit as st
import re
import math
import matplotlib.pyplot as plt
import joblib

from bs4 import BeautifulSoup


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove HTML
    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(" ")

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Keep apostrophes for contractions
    text = re.sub(
        r"[^a-zA-Z'\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD MODEL
# ============================================================

tfidf = joblib.load("tfidf.pkl")
svm_model = joblib.load("svm_model.pkl")


# ============================================================
# SENTIMENT PREDICTION FUNCTION
# ============================================================

def predict_sentiment(review):

    # --------------------------------------------------------
    # Split review into smaller parts
    # --------------------------------------------------------
    
    # Comma, period, semicolon, ! and ?
    sentences = re.split(
        r'[,.;!?]+',
        review
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    predictions = []
    valid_sentences = []

    # --------------------------------------------------------
    # Predict each sentence/part
    # --------------------------------------------------------

    for sentence in sentences:

        # Same preprocessing used during training
        cleaned = clean_text(sentence)

        # Skip empty text
        if not cleaned:
            continue

        # TF-IDF transformation
        sentence_tfidf = tfidf.transform(
            [cleaned]
        )

        # SVM prediction
        pred = svm_model.predict(
            sentence_tfidf
        )[0]

        predictions.append(pred)
        valid_sentences.append(sentence)

    # Use only sentences that were actually predicted
    sentences = valid_sentences

    # --------------------------------------------------------
    # COUNT SENTIMENTS
    # --------------------------------------------------------

    positive_count = predictions.count("Positive")
    negative_count = predictions.count("Negative")
    neutral_count = predictions.count("Neutral")

    # --------------------------------------------------------
    # OVERALL SENTIMENT
    # --------------------------------------------------------

    # Positive count greater than Negative count
    if positive_count > negative_count:

        overall_sentiment = "Positive"

    # Negative count greater than Positive count
    elif negative_count > positive_count:

        overall_sentiment = "Negative"

    # Equal Positive and Negative
    else:

        overall_sentiment = "Neutral"

    return (
        overall_sentiment,
        sentences,
        predictions
    )


# ============================================================
# GAUGE METER
# ============================================================

def create_gauge(sentiment):

    # --------------------------------------------------------
    # Position for each sentiment
    # --------------------------------------------------------

    positions = {
        "Negative": 0,
        "Neutral": 50,
        "Positive": 100
    }

    value = positions[sentiment]

    # Convert score to angle
    #
    # Negative = 180 degrees
    # Neutral  = 90 degrees
    # Positive = 0 degrees

    angle = 180 - (
        value * 1.8
    )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    # --------------------------------------------------------
    # DRAW SEMICIRCLE
    # --------------------------------------------------------

    theta = []
    meter_colors = []

    for i in range(101):

        value_i = i

        angle_i = 180 - (
            value_i * 1.8
        )

        theta.append(angle_i)

        # Color zones
        if value_i < 33.33:

            meter_colors.append("red")

        elif value_i < 66.66:

            meter_colors.append("gold")

        else:

            meter_colors.append("green")

    # --------------------------------------------------------
    # DRAW COLORED SEMICIRCLE
    # --------------------------------------------------------

    for i in range(
        len(theta) - 1
    ):

        x1 = 50 * math.cos(
            math.radians(theta[i])
        )

        y1 = 50 * math.sin(
            math.radians(theta[i])
        )

        x2 = 50 * math.cos(
            math.radians(theta[i + 1])
        )

        y2 = 50 * math.sin(
            math.radians(theta[i + 1])
        )

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=meter_colors[i],
            linewidth=15,
            solid_capstyle="round"
        )

    # ========================================================
    # NEEDLE
    # ========================================================

    needle_length = 40

    x = needle_length * math.cos(
        math.radians(angle)
    )

    y = needle_length * math.sin(
        math.radians(angle)
    )

    ax.plot(
        [0, x],
        [0, y],
        color="black",
        linewidth=5,
        solid_capstyle="round",
        zorder=4
    )

    # Center circle
    ax.scatter(
        0,
        0,
        s=300,
        color="black",
        zorder=5
    )

    # ========================================================
    # SCORE
    # ========================================================

    ax.text(
        0,
        -10,
        f"{value}%",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold"
    )

    # ========================================================
    # EMOJI
    # ========================================================

    if sentiment == "Positive":

        emoji_icon = "😊"

    elif sentiment == "Negative":

        emoji_icon = "😞"

    else:

        emoji_icon = "😐"

    ax.text(
        0,
        15,
        emoji_icon,
        ha="center",
        va="center",
        fontsize=28
    )

    # ========================================================
    # LABELS
    # ========================================================

    ax.text(
        -52,
        -5,
        "🔴 Negative",
        ha="center",
        fontsize=11
    )

    ax.text(
        0,
        53,
        "🟡 Neutral",
        ha="center",
        fontsize=11
    )

    ax.text(
        52,
        -5,
        "🟢 Positive",
        ha="center",
        fontsize=11
    )

    # ========================================================
    # GAUGE SETTINGS
    # ========================================================

    ax.set_xlim(
        -65,
        65
    )

    ax.set_ylim(
        -20,
        60
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

col1, col2 = st.columns(
    [1, 1]
)


# ============================================================
# LEFT SIDE
# ============================================================

with col1:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <h1 style="
            white-space: nowrap;
            font-size: 38px;
        ">
        📝 NLP Sentiment Analysis
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Enter a review and let the SVM model classify its sentiment."
    )

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    review = st.text_area(
        "Enter your review:",
        placeholder=(
            "Example: The phone is good, "
            "but the battery is bad."
        ),
        height=150
    )

    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )


# ============================================================
# ANALYZE SENTIMENT
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # CHECK EMPTY INPUT
    # --------------------------------------------------------

    if review.strip() == "":

        with col1:

            st.warning(
                "Please enter a review."
            )

    else:

        # ----------------------------------------------------
        # PREDICT SENTIMENT
        # ----------------------------------------------------

        (
            sentiment,
            sentences,
            predictions
        ) = predict_sentiment(
            review
        )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        with col2:

            # ------------------------------------------------
            # SENTIMENT RESULT
            # ------------------------------------------------

            st.subheader(
                "Sentiment Result"
            )

            # ------------------------------------------------
            # DISPLAY CLASSIFICATION
            # ------------------------------------------------

            if sentiment == "Positive":

                st.success(
                    "😊 Positive"
                )

            elif sentiment == "Negative":

                st.error(
                    "😞 Negative"
                )

            else:

                st.warning(
                    "😐 Neutral"
                )

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            st.subheader(
                "Sentiment Meter"
            )

            fig = create_gauge(
                sentiment
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # ------------------------------------------------
            # SENTIMENT COUNTS
            # ------------------------------------------------

            positive_count = predictions.count(
                "Positive"
            )

            negative_count = predictions.count(
                "Negative"
            )

            neutral_count = predictions.count(
                "Neutral"
            )

            st.subheader(
                "Sentiment Summary"
            )

            st.write(
                f"😊 Positive: {positive_count}"
            )

            st.write(
                f"😞 Negative: {negative_count}"
            )

            st.write(
                f"😐 Neutral: {neutral_count}"
            )

            # ------------------------------------------------
            # SENTENCE-LEVEL RESULTS
            # ------------------------------------------------

            if len(sentences) > 1:

                st.subheader(
                    "Sentence Analysis"
                )

            for sentence, prediction in zip(
                sentences,
                predictions
            ):

                # Remove quotation marks
                sentence = sentence.replace(
                    '"',
                    ''
                ).replace(
                    "'",
                    ''
                ).strip()

                # Remove unwanted symbols
                sentence = re.sub(
                    r'[^a-zA-Z0-9\s]',
                    ' ',
                    sentence
                )

                # Remove extra spaces
                sentence = re.sub(
                    r'\s+',
                    ' ',
                    sentence
                ).strip()

                # Skip empty/punctuation-only parts
                if (
                    not sentence
                    or not re.search(
                        r'[A-Za-z0-9]',
                        sentence
                    )
                ):
                    continue

                # ------------------------------------------------
                # EMOJI
                # ------------------------------------------------

                if prediction == "Positive":

                    emoji = "😊"

                elif prediction == "Negative":

                    emoji = "😞"

                else:

                    emoji = "😐"

                st.write(
                    f"**{sentence}** → "
                    f"{emoji} {prediction}"
                )
