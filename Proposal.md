# EECS 504 Project Proposal: TrackandMask

**Aidan Dempster, Andrew Krikorian, Audrey Douglas, Colin Fuelberth, Cody Sheltraw**  
adempst@umich.edu, akrik@umich.edu, aadougl@umich.edu, cfuel@umich.edu, cshel@umich.edu  

*March 24th, 2026*

---

## 1. What is your team name? Who is on the team?

**Team Name:** LGN10 Wanters  

**Team Members:**  
Aidan Dempster, Andrew Krikorian, Audrey Douglas, Colin Fuelberth, and Cody Sheltraw  

---

## 2. What is your project name?

**TrackAndMask**

---

## 3. What is your project description (4–6 sentences)?

Consistent video multi-person de-identification.

Our task will input a long form video with at least two people. We then create an id per unique person based on visual features, including those beyond just faces. These ids will be vectors in an embedding space where closeness represents similarity in the space of people. After doing so, we will segment faces and mask them with a colored overlay hashed by the unique id, leaving us with an anonymized video.

---

## 4. What do you expect to be able to demonstrate for your project by the end of the term?

We want to take in a long form video with a group of people throughout and output a robustly anonymized video (colored mask overlayed on faces) with a map of unique id's associated with each person.

---

## 5. What is the potential positive impact to society your project may enable?

This task allows for better sharing of sensitive data in a way that makes it easier to digest. Better communication of science that involves protected human data.

A relevant research project requires one to identify which children are performing which actions in order to build a profile of how individual children are acting and reacting to teaching. To enable both automated assigning of actions to unique person labels and IRB compliant anonymization while preserving unique visual identifiers.
