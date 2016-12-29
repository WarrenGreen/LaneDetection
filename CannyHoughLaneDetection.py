import cv2
import numpy as np

#input sources
capture = cv2.VideoCapture("highway_short.mp4")
fourcc = cv2.cv.FOURCC('F','M','P','4')

writer = None

#Create transformation matrices 
pts1 = np.float32([[575, 475],[696, 475],[220, 718],[1258, 718]])
pts2 = np.float32([[0,0],[200,0],[0,115],[200,115]])
M = cv2.getPerspectiveTransform(pts1,pts2)
MPrime = cv2.getPerspectiveTransform(pts2,pts1)

#Hough Line Transform params
minLineLength = 1
maxLineGap = 100

while True:
  ret, img = capture.read()
  rows,cols,ch = img.shape
  
  #Transform road perspective
  dst = cv2.warpPerspective(img,M,(200,115))
  gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
  
  #Canny Edge Detection
  edges = cv2.Canny(gray,100, 200,apertureSize = 3)
  
  #Hough Line Transform
  lines = cv2.HoughLinesP(image=edges,rho=5,theta=np.pi/500, threshold=10, minLineLength=minLineLength,maxLineGap=maxLineGap)
  
  #Draw Lines
  size = dst.shape[0], dst.shape[1], 3
  over = np.zeros(size, dtype=np.uint8)
  for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:
          cv2.line(over,(x1,y1),(x2,y2),(0,255,0),2)
  
  #Transform road back to original perspective
  over = cv2.warpPerspective(over,MPrime,(1277,719))
  over = cv2.resize(over, (cols, rows))
  fin = cv2.addWeighted( img , 0.5, over, 0.5, 0.0);
  
  #Show progress and write output video
  cv2.imshow('fin',fin)
  cv2.waitKey(5)
  if(writer == None):
    fin_row, fin_col, fin_d = fin.shape
    writer = cv2.VideoWriter("output.mp4", fourcc, 15, (fin_col, fin_row))
    writer.write(fin)
