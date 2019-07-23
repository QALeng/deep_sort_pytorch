# vim: expandtab:ts=4:sw=4
from config import  my_config

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
        初始状态的向量
    covariance : ndarray
        Covariance matrix of the initial state distribution.
        初始状态分布的协方差矩阵
    track_id : int
        A unique track identifier.
        id
    hits : int
        Total number of measurement updates.
        测量更新的总数
    age : int
        Total number of frames since first occurance.
        自第一次出现的总帧数
    time_since_update : int
        Total number of frames since last measurement update.
        自上次测量更新以来的帧总数。
    state : TrackState
        The current track state.
        当前状态
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
        特性缓存。在每次度量更新时，关联的特性
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,start_time,
                 feature=None, class_name="object"):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        #通过时间来删除对象，则不需要更新次数，所以注释掉了所有time_since_update
        # #更新次数
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

        #新增
        #类名
        self.class_name = class_name
        #检测到的时间
        self.start_time=start_time
        #离开标志位，为1表示还在，默认为零
        self.left_flag=1
        #第一次离开时间
        self.first_left_time=start_time

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1

        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        #当物体重新被检测到的时候，如果为0，则改为1
        if self.left_flag==0:
            self.left_flag=1

    ##这是原先的标记为检测的函数
    # def mark_missed(self):
    #     """Mark this track as missed (no association at the current time step).
    #     """
    #     if self.state == TrackState.Tentative:
    #         self.state = TrackState.Deleted
    #     elif self.time_since_update > self._max_age:
    #         self.state = TrackState.Deleted
    def mark_missed(self,now_time,left_time):
        """Mark this track as missed (no association at the current time step).
            #     """
        # if self.state == TrackState.Tentative:
        #     self.state = TrackState.Deleted
        # elif self.time_since_update > self._max_age:
        # if self.time_since_update > self._max_age:
        #     self.state = TrackState.Deleted


        #表示长时间离开将被删除，这里只是新增一个删除标志位
        if self.left_flag==0 and (now_time>self.first_left_time)>left_time:
            self.state=TrackState.Deleted
        #修改离开的标志位，只在第一次离开的时候修改      可能是出现了被遮挡的情况
        if self.left_flag==1:
            self.first_left_time=now_time
            self.state=TrackState.Tentative
            self.left_flag=0


    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
