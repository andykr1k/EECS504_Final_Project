# TrackAndMask
### Consistent video person de-identification.

## Components
### Dataset Creation:
We need a dataset of segmentations of individuals in videos. Each segment is assigned a unique ID. Associated with each segment ID we have a list of known positives and known negatives. Positives are other IDs for segments that are known to belong to the same individual and negatives are IDs for segments that are known to not belong to the same person.

Dataset generation from videos:
We can find positives by tracking individuals across frames using SAM. Until SAM loses the individual, all segmentations of that individual lie in the positive set. Negatives are all other segmentations of people in the scene that appear at the same time as the individual.

Other datasets to use:
Along with our generated dataset, we can use person reid datasets and reformat them into our dataset structure.
Try looking here https://github.com/NEU-Gou/awesome-reid-dataset.

#### *Dataset Structure:*
```
images/
  <img_id>.png
  <img_id>_segmentations.dat
  <img_id>_sidecar.json
```

#### *Segmentations Structure:*
Numpy long array of same shape as image. Values are the segment idx. -1 for background.
Segment idx is the index specified in the sidecar in the format "<img_id>_<segment_idx>".

#### *Sidecar Structure:*
```
[
    {
        "segment_id": str  # <img_id>_<segment_idx>
        "positives": [str]  # list of segment IDs
        "negatives": [str]  # list of segment IDs
    }
]
```
Positives are between images, negatives are mostly within images, but can also be between.



### Model Training:
We ingest the dataset and use contrastive losses (circle loss probably) to train an embedding model that learns to generate embeddings where being within a threshold of distance of each other implies that the segments belong to the same individual.

#### *Deliverable:*
Torch model wrapped in a class for easy inference. Should have an `embed` method that takes a PIL image and numpy bool array for the segmentation and returns the embedding.

### Inference:
**Clustering**: Assigning unique IDs to each segment.
Define a similarity matrix S where S[i, j] is the similarity between segment i and segment j. If i and j appear in the same scene at the same time, they cannot be the same person so set similarity low. Otherwise, set similarity to the cosine similarity of their embeddings. Then cluster the segments using agglomerative clustering.

**Masking**: Segment faces in the video using a face segmentation model. Find overlap with a SAM segmentation for the individual. Assign the segmentation to one of the clusters. Color the face using the cluster ID.