function iou = similarityIoU(tracks, detections)
% Calculate the Intersection over Union similarity between tracks and
% detections

states = [tracks.State];
bbstate = helperBBMeasurementFcn(states); % Convert states to [x, y, w, h] for bboxOverlapRatio
bbmeas = vertcat(detections.Measurement);
bbmeas = helperBBMeasurementFcn(bbmeas')';
iou = bboxOverlapRatio(bbstate', bbmeas); % Bounding boxes must be concatenated vertically
end