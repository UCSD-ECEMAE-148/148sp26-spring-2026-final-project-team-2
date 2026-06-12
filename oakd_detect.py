#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time

TAG_SIZE = 0.162

half = TAG_SIZE/2.0
obj_pts = np.array([
    [-half, half, 0],
    [half, half ,0],
    [half, -half, 0],
    [-half,-half, 0],
], dtype=np.float64)

# FULL_RES = (4000, 3000) # 12MP
# FULL_RES = (1920, 1080)
FULL_RES = (640,480)
PREVIEW_SIZE = (960, 540) # 1/3 of 12MP, to preserve bandwidth

with dai.Pipeline() as pipeline:
    hostCamera = pipeline.create(dai.node.Camera).build()
    aprilTagNode = pipeline.create(dai.node.AprilTag)
    outputCam = hostCamera.requestOutput(FULL_RES)
    outputCam.link(aprilTagNode.inputImage)
    outQueue = aprilTagNode.out.createOutputQueue()

    device = pipeline.getDefaultDevice()
    calib = device.readCalibration()
    M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,FULL_RES[0],FULL_RES[1])
    D = calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

    camera_matrix = np.array(M, dtype=np.float64)
    dist_coeffs = np.array(D, dtype=np.float64)

    print(M,D)
    # We use ImageManip instead of creating a new smaller output because of the syncing,
    # ATM, AprilTags don't contain timestamps, so we can't sync them with frames
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(PREVIEW_SIZE[0], PREVIEW_SIZE[1], dai.ImageManipConfig.ResizeMode.STRETCH)
    manip.setMaxOutputFrameSize(2162688)
    outputCam.link(manip.inputImage)
    frameQ = manip.out.createOutputQueue()

    color = (0, 255, 0)
    startTime = time.monotonic()
    counter = 0
    fps = 0.0
    pipeline.start()

    while pipeline.isRunning():
        aprilTagMessage = outQueue.get()
        assert(isinstance(aprilTagMessage, dai.AprilTags))
        aprilTags = aprilTagMessage.aprilTags

        counter += 1
        currentTime = time.monotonic()
        if (currentTime - startTime) > 1:
            fps = counter / (currentTime - startTime)
            counter = 0
            startTime = currentTime

        def rescale(p: dai.Point2f):
            return (int(p.x / FULL_RES[0] * PREVIEW_SIZE[0]),
                    int(p.y / FULL_RES[1] * PREVIEW_SIZE[1]))

        passthroughImage: dai.ImgFrame = frameQ.get()
        frame = passthroughImage.getCvFrame()
        for tag in aprilTags:
            # topLeft = rescale(tag.topLeft)
            # topRight = rescale(tag.topRight)
            # bottomRight = rescale(tag.bottomRight)
            # bottomLeft = rescale(tag.bottomLeft)

            # center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))

            # cv2.line(frame, topLeft, topRight, color, 2, cv2.LINE_AA, 0)
            # cv2.line(frame, topRight,bottomRight, color, 2, cv2.LINE_AA, 0)
            # cv2.line(frame, bottomRight,bottomLeft, color, 2, cv2.LINE_AA, 0)
            # cv2.line(frame, bottomLeft,topLeft, color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(tag.id)
            # cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            # cv2.putText(frame, f"fps: {fps:.1f}", (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
            img_pts = np.array([
                [tag.topLeft.x, tag.topLeft.y],
                [tag.topRight.x, tag.topRight.y],
                [tag.bottomRight.x, tag.bottomRight.y],
                [tag.bottomLeft.x, tag.bottomLeft.y],
            ], dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok:
                x, y, z = tvec.flatten()
                dist = np.linalg.norm(tvec)
                print(
                    f"[FPS {fps:5.1f}] "
                    f"Tag {tag.id:3d} | "
                    f"dist: {dist:.3f}m | "
                    f"x: {x:+.3f} y: {y:+.3f} z: {z:.3f}"
                )
        print(fps)
        # cv2.imshow("detections", frame)

        if cv2.waitKey(1) == ord("q"):
            break

