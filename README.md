# Email Triage System

A lightweight email prioritization system that filters important emails from hundreds of daily messages and provides automated reply suggestions with heuristic-based reasoning logs.

## Overview

Managing large volumes of emails is challenging, especially when important messages are buried among newsletters, notifications, and low-priority communications. This project builds a personalized email triage system that learns a user's reading behavior and prioritizes incoming emails based on historical interaction patterns and content relevance.

The system assigns a priority score to every email and surfaces the most important messages. Additionally, it generates automated reply drafts for selected emails using a Large Language Model (LLM).
<img width="1915" height="833" alt="image" src="https://github.com/user-attachments/assets/50ea49c9-b0cd-420e-bafe-c65e82d76e1c" /><img width="1302" height="822" alt="image" src="https://github.com/user-attachments/assets/8a06a1b4-61ea-4602-b016-62a20051eb30" /><img width="1390" height="657" alt="image" src="https://github.com/user-attachments/assets/7e4abac3-0749-48ba-942c-5d3ea22809bf" />




## Dataset

The dataset consists of approximately 10–11 months of Gmail data collected through the Gmail API from a college email account.

### Extracted Features

* Subject
* Snippet
* Sender Name
* Recipient Name
* Thread ID
* Date
* CC Count
* Attachment Indicator

### Why Not Use Email Body?

The email body was intentionally excluded because:

* Subject and snippet generally capture 80–90% of the email's intent.
* Processing only metadata significantly reduces inference latency.
* Lower storage and computational requirements.
* Faster real-time prioritization.

## Feature Engineering

The system combines social, behavioral, temporal, and textual signals to determine email importance.

### 1. Social Features

These features model the relationship between the sender and recipient.

#### Intuition

If a user frequently receives and reads emails from a particular sender, future emails from that sender are more likely to be important.

**Example**

Emails from your organization, professors, placement cell, or project mentors.

#### Feature

**Sender Percentage**

Sender Percentage = Read Emails from Sender / Total Emails Received from Sender​

This feature is computed using historical read-email statistics and represented as a percentage.

---

### 2. Thread Features

These features capture conversation engagement.

#### Intuition

If a user actively participates in a thread by replying multiple times, future emails in the same thread are likely to be important.

**Example**

Ongoing discussions with professors, teammates, recruiters, or managers.

#### Feature

**Thread Count**

Number of times the user has interacted with a specific thread, computed from historical read emails.

---

### 3. Recent Interest Features

These features capture evolving user interests.

#### Intuition

Users often focus on particular topics for limited periods. Emails matching recently consumed topics should receive higher priority.

**Example**

If a user has recently read many emails related to internships, placements, interviews, or research opportunities, new emails on similar topics are likely important.

#### Feature Generation

1. Collect read emails from the previous 30 days.
2. Extract words from subject and snippet.
3. Select words above the 95th percentile frequency threshold.
4. Check whether the current email contains these words.

#### Feature

**Contains Top Words**

Boolean feature indicating whether the email matches recent interests.
<img width="1255" height="677" alt="image" src="https://github.com/user-attachments/assets/01e5695b-e878-4497-82a8-9031490c1ad4" />

---

### 4. Content Features

Textual information remains one of the strongest indicators of email importance.

#### Processing Pipeline

1. Concatenate Subject and Snippet.
2. Apply TF-IDF vectorization.
3. Generate sparse text representations.
4. Apply PCA for dimensionality reduction.
5. Retain components explaining 90% of total variance.

This produces compact semantic representations while reducing noise and computational cost.

---

### 5. Additional Features

Additional metadata used by the model:

* CC Count
* Attachment Presence
* Temporal information derived from email timestamps

## Model Training

The project follows an incremental modeling strategy.

### Baseline Model

**Logistic Regression**

* Fast and interpretable
* Provides a strong baseline for binary classification

### Advanced Model

**Random Forest Classifier**

* Captures non-linear relationships
* Handles mixed feature types effectively
* Better suited for behavioral and interaction-based patterns

## Problem Formulation

The task is modeled as a binary classification problem:

### Classes

* **Read Email (Important)**
* **Unread Email (Less Important)**

The model learns from historical user behavior to predict the likelihood that a newly received email will be read.

## Model Evaluation

### Evaluation Metric

**Precision**

Precision is prioritized because surfacing relevant emails as low-priority creates a poor user experience.

Precision = TP / (TP + FP)

where:

* TP = True Positives
* FP = False Positives

A higher precision ensures that emails marked as important are genuinely valuable to the user.
<img width="557" height="443" alt="image" src="https://github.com/user-attachments/assets/0df20a08-ddb6-44ee-b039-cddeecdf1fd8" />


## Priority Scoring

Although the underlying task is binary classification, the predicted probability is used as a continuous priority score.

### Priority Score


Priority_Score = P(Read \mid Email)

Emails are ranked in descending order of this score.

This allows:

* Personalized inbox ranking
* Top-N important email retrieval
* Dynamic prioritization thresholds

## Heuristic Reason Logging

To improve transparency and explainability, the system stores heuristic reasons behind prioritization decisions.

Example explanations:

* Frequent sender interaction
* High thread engagement
* Matches recent interests
* Contains attachment
* Similar to previously read emails

These logs help users understand why an email received a high priority score.

## Automatic Reply System

Once high-priority emails are identified:

1. The email body is fetched on demand.
2. The content is sent to a Gemini-based LLM.
3. The model generates a context-aware reply draft.
4. The user can review, edit, and send the response.

### Benefits

* Reduced email management time
* Faster response generation
* Consistent communication quality
* Improved productivity

## Pipeline

```text
Gmail API
    |
    v
Email Metadata Extraction
    |
    v
Feature Engineering
    |
    v
TF-IDF + PCA
    |
    v
Classification Model
(Logistic Regression / Random Forest)
    |
    v
Priority Score Generation
    |
    v
Email Ranking
    |
    +--------------------+
    |                    |
    v                    v
Reason Logging     Gemini LLM
                         |
                         v
                    Automatic Reply Generation
```
### Why Random Forest Was Chosen Over Logistic Regression

Although Logistic Regression provided a strong baseline, Random Forest was ultimately selected as the final model due to its ability to better capture non-linear relationships between user behavior and email importance.

An analysis of feature importance revealed a key difference between the two models:

* **Logistic Regression** primarily relied on the **Thread Count** feature, indicating that ongoing conversations were the strongest signal it used for classification.
* **Random Forest** effectively leveraged both **Social Features** (Sender Percentage) and **Thread Features**, allowing it to capture a broader range of user interaction patterns.

This behavior aligns well with real-world email usage. An email can be important not only because it belongs to an active conversation thread, but also because it originates from a sender with whom the user frequently interacts. By considering both signals simultaneously, Random Forest produces more personalized and reliable prioritization decisions.

As a result, Random Forest demonstrated a better understanding of user-specific communication patterns and was chosen as the final model for generating email priority scores.
<img width="1246" height="407" alt="image" src="https://github.com/user-attachments/assets/73953dc1-e075-4d89-9466-0b6bee938d63" />
<img width="691" height="328" alt="image" src="https://github.com/user-attachments/assets/cf8f94e0-6125-43bd-86d3-cd9706015de1" />

## Future Improvements

### 1. Time-Based Priority Modeling

Currently, emails are labeled based on whether they are read or not. However, the time taken by a user to read an email can provide a stronger signal about its actual importance.

For example, emails that are opened within a few hours of arrival are generally more important than emails that are read after 10–15 days. Delayed interactions may indicate casual browsing rather than genuine urgency.

Future versions of the system can incorporate a configurable time window (e.g., 24 hours) and assign higher importance to emails read within that period. Emails outside the window can be filtered or assigned lower weights during training, improving both prioritization quality and inference speed.

### 2. User Feedback Loop

The current system relies solely on historical reading behavior. A feedback mechanism can make the model more personalized over time.

* If a user approves a recommended email, its predicted priority score can be reinforced.
* If a user dismisses or dislikes a recommendation, the score can be adjusted downward (e.g., `1 - priority_score`).

As feedback data accumulates, the problem can be reformulated from binary classification to a regression task, where the model learns a continuous measure of email importance directly from user preferences. This would enable finer-grained ranking and improved personalization.

### 3. Domain-Specific Open-Source LLMs for Reply Generation

The current automatic reply system uses Gemini to generate email responses. While effective, the generated replies can sometimes be generic.

Future iterations can leverage fine-tuned open-source language models trained on email communication datasets. Such models can generate more context-aware, personalized, and domain-specific responses while also providing greater control over deployment, privacy, and customization.

Potential improvements include:

* Personalized writing style adaptation
* Organization-specific response generation
* On-device or self-hosted deployment
* Reduced inference costs
* Better contextual understanding of email conversations

## Tech Stack

* Python
* Gmail API
* Scikit-Learn
* Pandas
* NumPy
* TF-IDF
* PCA
* Logistic Regression
* Random Forest
* Gemini API

## Reference Paper
https://research.google/pubs/the-learning-behind-gmail-priority-inbox

