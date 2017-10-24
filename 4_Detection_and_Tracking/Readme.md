Goal of this program:

1. Detect the face in the first frame of the movie using pre-trained Viola-Jones detector
2. Track the face throughout the movie using:
   CAMShift
   Particle Filter
   Face detector + Kalman Filter (always run the kf.predict(), and run kf.correct() when you get a new face detect
