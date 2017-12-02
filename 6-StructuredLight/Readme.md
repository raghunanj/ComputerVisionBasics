Goal : To reconstruct a scene from multiple structured light scannings of it.

1. Calibrate projector with the “easy” method.             <br />
  a. Use ray-plane intersection                            <br />
  b. Get 2D-3D correspondence and use stereo calibration.  <br />
  
2. Get the binary code for each pixel.                     
3. Correlate code with (x,y) position - a "codebook" from binary code -> (x,y) 
4. 2D-2D correspondence                                    <br />
  a. Perform stereo triangulation (existing function) to get a depth map
