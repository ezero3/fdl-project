# Project Requirements and Delivery Rules

This document records the official organizational, submission, presentation, and technical constraints for the Foundations of Deep Learning project in the 2025/2026 academic year.

## Key dates

| Event | Date and time |
|---|---|
| Project submission deadline | Tuesday, 8 September 2026, 23:59 (Italy time) |
| Official exam and presentation date | Tuesday, 15 September 2026 |

The code and presentation must be submitted through eLearning one week before the official exam date. Small changes may be made during the final week, but the required material must already have been submitted by the deadline. Depending on the number of registered students, presentations may be divided across two days; the professors will communicate any split.

## Group organization

- Each group must contain two or three students.
- Students organize their groups through eLearning and should communicate the composition as early as possible.
- One group representative is responsible for the eLearning submission.

## Required deliverables

The eLearning submission must include:

1. presentation slides in PDF format;
2. the developed Python code;
3. trained model files or saved weights, such as `.h5` or `.keras` files.

If trained models exceed the upload limit, the eLearning submission may contain a Google Drive link for those model files only.

### Submission restrictions

- Do not submit project files or links by email.
- Do not submit the slides through a Google Drive link.
- Do not submit the code through a Google Drive link.
- External links are permitted only for trained models when direct upload is impractical because of file size.
- The deadline must still be met through the official eLearning submission.

## Presentation requirements

- The oral presentation and all slides must be in English.
- Each member should speak for approximately five minutes.
- A two-person group should plan approximately 10 minutes and 10 slides.
- A three-person group should plan approximately 15 minutes and 15 slides.
- Additional time is reserved for individual questions.

The recommended narrative is:

1. introduce the group members;
2. define the problem clearly;
3. describe and analyze the available data;
4. explain the proposed deep learning solution, architecture, training procedure, and design choices;
5. discuss positive and negative results using quantitative and qualitative evidence;
6. present alternative approaches, including relevant unsuccessful experiments.

Items 4–6 may be reordered when a different sequence produces a clearer presentation.

## Dataset and modeling constraints

- The project must use a dataset proposed by the professors, unless an external dataset is approved in advance.
- A request to use an external dataset must report its number of instances, dimensionality, number of classes, proposed task, and other relevant properties.
- Tasks marked in orange in the official topic document were not covered during the course and may be selected at the students' own risk.
- If a dataset is too large, subsampling is allowed and must be explicitly discussed in the presentation.
- Global random sampling is preferred when practical.
- Removing classes only to simplify the task is discouraged and requires a clear justification.

## Use of pretrained models

Pretrained models are allowed, but the solution cannot rely exclusively on them. The group must develop and train a meaningful component. Acceptable strategies include:

- extracting pretrained features or embeddings and training original neural layers on top;
- developing a model from scratch and comparing it with a pretrained solution;
- transferring knowledge from an existing model to a different task.

## Academic integrity

Anti-plagiarism software will be used to scan submitted solutions. The implementation and submitted materials must represent the group's own work, with external material cited appropriately.

## Repository and delivery structure

The repository is organized to keep development artifacts and final deliverables separate:

| Path | Intended content |
|---|---|
| `data/` | Local raw and processed dataset files; large data should not be committed |
| `docs/` | Documentation and project requirements |
| `notebooks/` | Exploration, preprocessing, and experiment notebooks |
| `specs/` | Technical specifications and experiment definitions |
| `tests/` | Automated tests for reusable project code |
| `trained-models/` | Saved weights and trained models; use external storage if necessary |
| `presentation/` | Presentation planning and supporting material |
| `presentation-slides/` | Slide sources and the final PDF export |
| `explanation/` | Detailed explanations, results, and qualitative analysis |

Before submission, verify that the final PDF slides, Python code, and model artifacts or permitted model link are complete and accessible from the eLearning submission.
