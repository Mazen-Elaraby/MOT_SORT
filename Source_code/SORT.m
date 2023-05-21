%ECE359 Signal Processing for Multimedia Major Task
%Implementing the SORT Algorithm (Simple Online and Real-time Tracking)
%for the Multi-object tracking (MOT) Problem 

%Downloading Pedestrian Tracking Video

datasetname="PedestrianTracking";
videoURL = "https://ssd.mathworks.com/supportfiles/vision/data/PedestrianTrackingVideo.avi";
if ~exist("PedestrianTrackingVideo.avi","file")
    disp("Downloading Pedestrian Tracking Video (35 MB)")
    websave("PedestrianTrackingVideo.avi",videoURL);
end

%file contains detections generated from a people detector using aggregate channel features (ACF).
load("PedestrianTrackingACFDetections.mat","detections");

%detection noise is zero-mean Gaussian with the following covariance matrix
R = diag([1, 1, 10, 1]);

%converting detection measurments to the tracking state vector 
% in the objectDetection (Sensor Fusion and Tracking Toolbox) format.
convertedDetsACF = helperConvertBoundingBox(detections,R);


IoUmin = 0.05; %a threshold for the association of detections to tracks 

tracker = trackerGNN(FilterInitializationFcn=@helperInitcvbbkf,...
    HasCostMatrixInput=true,...
    AssignmentThreshold= -IoUmin);

%Defining Track Maintenance
TLost = 3;
tracker.ConfirmationThreshold=[2 2];
tracker.DeletionThreshold=[TLost TLost];



% Load and convert YOLOv4 detections
load("PedestrianTrackingYOLODetections.mat","detections");
convertedDetsYOLO = helperConvertBoundingBox(detections, R);
detectionScoreThreshold = -1;
showAnimation = true;

yoloSORTTrackLog = helperRunSORT(tracker, convertedDetsYOLO, detectionScoreThreshold, showAnimation);
acfSORTTrackLog = helperRunSORT(tracker, convertedDetsACF, detectionScoreThreshold, showAnimation);
%Evaluate SORT with the CLEAR MOT Metrics
threshold = 0.1;
tcm = trackCLEARMetrics(SimilarityMethod ="IoU2d", SimilarityThreshold = threshold);

acfTrackedObjects = repmat(struct("Time",0,"TrackID",1,"BoundingBox", [0 0 0 0]),size(acfSORTTrackLog));
for i=1:numel(acfTrackedObjects)
    acfTrackedObjects(i).Time = acfSORTTrackLog(i).UpdateTime;
    acfTrackedObjects(i).TrackID = acfSORTTrackLog(i).TrackID;
    acfTrackedObjects(i).BoundingBox(:) = helperBBMeasurementFcn(acfSORTTrackLog(i).State(1:4));
end

yoloTrackedObjects = repmat(struct("Time",0,"TrackID",1,"BoundingBox", [0 0 0 0]),size(yoloSORTTrackLog));
for i=1:numel(yoloTrackedObjects)
    yoloTrackedObjects(i).Time = yoloSORTTrackLog(i).UpdateTime;
    yoloTrackedObjects(i).TrackID = yoloSORTTrackLog(i).TrackID;
    yoloTrackedObjects(i).BoundingBox(:) = helperBBMeasurementFcn(yoloSORTTrackLog(i).State(1:4));
end

load("PedestrianTrackingGroundTruth.mat","truths");
acfSORTresults = evaluate(tcm, acfTrackedObjects, truths);
yoloSORTresults = evaluate(tcm, yoloTrackedObjects, truths);

allResults = [table("ACF+SORT",VariableNames = "Tracker") , acfSORTresults ; ...
    table("YOLOv4+SORT",VariableNames = "Tracker"), yoloSORTresults];

disp(allResults);

