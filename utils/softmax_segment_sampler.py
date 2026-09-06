### a function that takes as input a list of frames and computes the weighted frobenius norm of the frames
### then computes the softmax of the frames
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def compute_frobenius_difference(frames):

    """
    Computes the sum of normalized Frobenius norms of differences between consecutive frames.

    This function calculates the Frobenius norm of the difference between each pair of
    consecutive matrices (frames), normalizes each value by dividing it with `(1 + max norm)`,
    and returns the sum of the normalized differences.

    Args
    ----------
    frames : list or array-like of ndarray
        A sequence of 2D arrays (matrices), where each matrix represents a frame.

    Returns
    -------
    float
        The sum of the normalized Frobenius norms of differences between consecutive frames.
    
    Notes
    -----
    - The Frobenius norm is computed as the square root of the sum of the absolute squares
      of the matrix elements.
    - Normalization helps scale the differences relative to the maximum difference observed.
    """

    differences = []

    for i in range(len(frames) - 1):
        # Compute the difference between consecutive frames
        diff = frames[i + 1] - frames[i]
        # Compute Frobenius norm of the difference
        frobenius_norm = np.linalg.norm(diff, ord='fro')  # 'fro' specifies Frobenius norm
        differences.append(frobenius_norm)
    differences = [d /(1+ max(differences)) for d in differences]
    return np.sum(differences)


def softmax(scores):
    """
    Compute the softmax scores of a list.

    Args:
        scores (list of float): List of numerical scores.

    Returns:
        list of float: List of softmax scores corresponding to the input scores.
    """
    #  Compute the exponential of each score (with numerical stability).
    max_score = max(scores)  # Max value for numerical stability
    exp_scores = [math.exp(score - max_score) for score in scores]
    
    #  Compute the sum of the exponentials.
    sum_exp_scores = sum(exp_scores)
    
    #  Compute the softmax scores.
    softmax_scores = [exp_score / sum_exp_scores for exp_score in exp_scores]
    
    return softmax_scores

def load_video_clips(path):
    """
    Loads a video from the specified path, extracts grayscale frames, and generates clips.

    This function reads a video file frame by frame, converts each frame to grayscale,
    stores the frames in a list, and then generates clips using the `generate_clips` function.

    Args
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    list
        A list of video clips generated from the grayscale frames.

    Notes
    -----
    - Frames are converted to grayscale using OpenCV's `cv2.cvtColor` with `cv2.COLOR_BGR2GRAY`.
    - The `generate_clips` function is assumed to handle segmentation of the frame list into clips.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        #frames.append(frame)
    cap.release()
    clips = generate_clips(frames)
    return clips

def generate_clips(frames):
    """
    Splits a list of video frames into fixed-length clips, padding if necessary.

    This function divides a sequence of frames into smaller clips of a fixed length.
    If the final clip contains fewer frames than the desired length, it is padded
    by repeating the last frame to maintain consistency.

    Parameters
    ----------
    frames : list
        A list of video frames (e.g., grayscale or color images represented as arrays).

    Returns
    -------
    list of lists
        A list where each element is a clip (a list of `clip_length` frames).

    Notes
    -----
    - The default clip length is 8 frames.
    - Padding is applied to the final clip if there are not enough frames to meet
      the required clip length.
    """
    clips = []
    total_frames = len(frames)
    #This is just an example ..!
    clip_length=8

    for start in range(0, total_frames,clip_length):
        end = start + clip_length
        clip = frames[start:end]
        # lets also account for inconsistency in length of frames
        if len(clip)<clip_length:
        # we pad the clip with the last frame
            pad = [clip[-1]] * (clip_length - len(clip))
        # extend clip to reach clip length
            clip.extend(pad)
        clips.append(clip)
    return clips



def soft_max_probabilities(path):
    """
    Computes softmax probabilities based on Frobenius differences of video clips.

    This function loads video clips from the given path, computes the Frobenius-based
    difference score for each clip using `compute_frobenius_difference`, and then applies
    the softmax function to convert the scores into a probability distribution.

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    numpy.ndarray
        A 1D array of softmax probabilities corresponding to each video clip.

    Notes
    -----
    - The function uses `load_video_clips` to extract and segment frames.
    - `compute_frobenius_difference` measures the temporal variation within each clip.
    - Higher Frobenius difference implies greater temporal change, and the softmax
      converts these differences into relative importance weights.
    """
    clips=load_video_clips(path)
    #list to append the weighted frobenius norm of each clip
    weight_frob=[]
    for frames in clips:
        weight_frob.append(compute_frobenius_difference(frames))

    return softmax(weight_frob)


