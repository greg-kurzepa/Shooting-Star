# A Cell Comet Assay Analysis Tool

Uses OpenCV to process comet assay images and automatically extract metrics, allowing manual exclusion of outlier cells and adjustment of cell regions.

Easily exclude outlier cells
<img width="1058" height="595" alt="example1" src="https://github.com/user-attachments/assets/7b0b4560-c465-4f9f-b1d8-5099e1649d23" />
Easily change the threshold between cell and comet regions
<img width="1058" height="595" alt="example2" src="https://github.com/user-attachments/assets/f609729d-f883-40b6-91f7-fdef1330df85" />

The following metrics are automatically collected in csv after processing, for each cell in each image:
- cell body area in pixels
- cell comet area in pixels
- body intensity
- comet intensity
- comet percentage intensity

Note, this is a work in progress. Sometimes cells in input images are erroneously ignored. Configuring the processing parameters is necessary.
Using Python 3.10.0 (see requirements.txt for dependencies).
