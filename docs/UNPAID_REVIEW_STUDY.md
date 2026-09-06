# Volunteer image review

AeroSynth-Eval needs people to judge its aircraft inspection images. Public
labels from AGDD and GenAI-Bench are useful checks, but they are not reviews of
these generated images. The first step is a 12-case development pilot with two
independent volunteers.

## Finding reviewers for free

Contact AIAA student branches, university aerospace clubs, aviation maintenance
schools, nondestructive inspection groups, and contributors to open aircraft
inspection projects. Zooniverse also has a free project builder and a volunteer
community, although a public project must pass Zooniverse review first.

Ask for relevant experience, not job titles. A graduate student who has worked
with aircraft structures may be a better reviewer than a senior engineer who
has never inspected imagery. Record the qualification in broad terms and keep
names and contact details out of Git.

## Build the pilot files

```bash
python scripts/build_unpaid_review_pilot.py \
  --output-dir outputs/volunteer_review_pilot
```

Send one CSV to each volunteer with `docs/ANNOTATION_GUIDE.md` and the referenced
development images. They must work separately. The pilot checks whether the
rubric is understandable; it is not protected-test evidence and must not be
described as a completed human study.

## Outreach note

> I'm testing an open-source tool that checks synthetic aircraft inspection
> images. I need two volunteers with aerospace, aircraft maintenance, or visual
> inspection experience to rate 12 development examples independently. It
> should take about 30 minutes. There is no payment. I can credit you as a
> contributor, use a pseudonym, or keep your review anonymous. You can stop at
> any point.

After both files return, run the existing ingestion and agreement commands.
Resolve disagreements with a third volunteer who has not seen the model score.
Do not fill missing responses, infer reviewer decisions, or count the dataset's
original labels as new independent reviews.
