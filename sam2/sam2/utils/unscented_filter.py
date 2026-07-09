import numpy as np
import scipy.linalg

class UnscentedKalmanFilter(object):
    """
    An extension of the Kalman filter for tracking bounding boxes in image 
    space.

    The 12-dimensional state space

        x, y, a, h, x_v, y_v, a_v, h_v, x_a, y_a, a_a, h_a
    
    this contains the cartesian coordaintes of the centre of the bounding box,
    the aspect ratio a, and height h, with their respective velocities and
    accelerations.
    
    """
    def __init__(self, alpha, beta, kappa, n_x, n_m):
        # naming convention that indicates that it shouldn't changed externally
        self._n_dim = 4
        self._delta_t = 1
        
        # hyper parameters
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._lamda = (alpha ** 2) * (kappa + n_x) - n_x
        
        self._n_x = n_x # number of states
        self._n_m = n_m # number of measured states
        self._n_s = (2 * n_x) + 1 # number of sigma points

        # pos, vel are the same as KF
        self._std_weight_pos = 1. / 20
        self._std_weight_vel = 1. / 160
        self._std_weight_acc = 1. / 160
        
    def initiate(self, z):
        """
        Creates an initial state vector, and estimate covariance

        Parameters
        ----------
        z : ndarray of shape (n_m, 1)
            Bounding box location (x,y,a,h)
        
        Returns
        -------
        x : ndarray of shape (n_x, 1)
            Initial state vector.
        P : ndarray of shape (n_x, n_x)
            Initial covariance matrix.
        
        Notes
        -----
        where:
            n_x : number of states in state vector
            n_m : number of measured states 
        """

        pos = z
        vel = np.zeros_like(z)
        acc = np.zeros_like(z)

        # initial state vector 
        # [x, y, a, h, 0, 0, 0, 0, 0, 0, 0, 0]
        x = np.r_[pos, vel, acc]

        # TODO : Finetune Standard deviation

        # estimate covariance
        std_pos = [ 
            2 * self._std_weight_pos * x[3],
            2 * self._std_weight_pos * x[3],
            1e-2,
            2 * self._std_weight_pos * x[3],
        ]

        std_vel = [ 
            10 * self._std_weight_vel * x[3],
            10 * self._std_weight_vel * x[3],
            1e-5,
            10 * self._std_weight_vel * x[3],
        ]

        std_acc = [ 
            10 * self._std_weight_acc * x[3],
            10 * self._std_weight_acc * x[3],
            1e-5,
            10 * self._std_weight_acc * x[3],
        ]
        
        std = np.r_[std_pos, std_vel, std_acc]
        P = np.diag(np.square(std))

        return x, P
    
    def _scaled_sigma_point_generation(self, x, P):
        """
        Performs Van der Merwe's Scaled Sigm point generation 
        
        Parameters
        ----------
        x : ndarray of shape (n_x, 1)
            Estimated system state vector at time k
        P : ndarray of shape (n_x, n_x)
            Current uncertainty covariance matrix at time k
        Returns
        -------
        X_sigma : ndarray of shape (n_s, n_x)
            Generated sigma points
        W_c : ndarray of shape (n_s, 1)
            Weights for covariance
        W_m : ndarray of shapoe (n_s, 1)
            Weights for mean

        Notes
        -----
        where:
            n_x : number of states in state vector
            n_s : number of sigma points 
            k : time step
        """

        # calculation of state and covariance weights
        W_c = np.zeros(self._n_s)
        W_m = np.zeros(self._n_s)

        w = self._lamda / (self._n_x + self._lamda)
        W_m[0] = w

        w = w + 1 - (self._alpha ** 2) + self._beta
        W_c[0] = w

        w = 1 / (2 * (self._n_x + self._lamda))
        W_m[1:] = w
        W_c[1:] = w

        # calculation of sigma points
        X_sigma = np.zeros((self._n_s, self._n_x))
        X_sigma[0] = x

        try: 
            S = np.linalg.cholesky((self._n_x + self._lamda) * P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky((self._n_x + self._lamda) * (P + np.eye(self._n_x) * 1e-9))

        for idx in range(self._n_x):
            X_sigma[idx + 1] = x + S[:, idx]
            X_sigma[idx + self._n_x + 1] = x - S[:, idx]

        return X_sigma, W_m, W_c
            
    def _f_constant_acceleration(self, x):
        """
        State transition function for constant acceleration motion model.

        State vector:
            x = [x_position     ] 
                [y_position     ] 
                [aspect_ratio   ] 
                [height         ] 
                [x_velocity     ] 
                [y_velocity     ]
                [a_velocity     ]
                [h_velocity     ]
                [x_acceleration ]
                [y_acceleration ]
                [a_acceleration ]
                [h_acceleration ]

        System Dynamics
            Assume that moving object is able to move at a constant acceleration.
            pos_k+1 = pos_k + (delta_t * vel_k) + (1/2 * delta_t^2 * acc_k)
            acc_k+1 = acc_k
            vel_k+1 = vel_k + delta_t * acc_k
            Time Step of 1 Second delta_t

        Parameters
        ----------
            x : ndarray of shape (n_x, 1)
                Estimated system state vector at time k
            delta_t : constant
                Time step since last measurement/prediction

        Returns
        -------
            F : ndarray of shape (n_x, 1)
                Sigma point projected through state transition function

        Notes
        -----
            where:
                n_x : number of states in state vector
                k : time step
        """

        F = np.zeros((self._n_x))

        for idx in range(4):
            # indexes of each state
            v_idx = idx + 4
            a_idx = idx + 8
            
            F[idx] = x[idx] + (self._delta_t * x[v_idx]) + ((1/2 * (self._delta_t ** 2)) * x[a_idx]) # state
            F[v_idx] = x[v_idx] + (self._delta_t * x[a_idx]) # velocity
            F[a_idx] = x[a_idx] # acceleration

        return F
    
    def _project_sigma_points(self, X_sigma):
        """
        Project sigma points according to the process model forming 
        a new set of sigma points. 
        
        Parameters
        ----------
            X_sigma : ndarray of shape (n_s, n_x)
                Sigma points 
            delta_t : constant
                Time step since last measurement/prediction

        Returns
        -------
            Y_sigma : ndarray of shape (n_s, n_x)
                Sigma point projected through state transition function

        Notes
        -----
            where:
                n_x : number of states in state vector
                n_s : number of sigma points 
                k : time step
        """
        Y_sigma = np.zeros((self._n_s, self._n_x))
        
        for idx in range(self._n_s):
            Y_sigma[idx] = self._f_constant_acceleration(X_sigma[idx])
                    
        return Y_sigma

    def _unscented_transform(self, Y_sigma, W_c, W_m, Q):
        """
        Performs an unscented transform onto the projected sigma points.

        Parameters
        ----------
            Y_sigma : ndarray of shape (n_s, n_x)
                Projected Sigma points 
            W_c : ndarray of shape (n_s, 1)
                Weights for covariance
            W_m : ndarray of shape (n_s, 1)
                Weights for mean
            Q : ndarray of shape (n_x, n_x)
                Covariance Matrix

        Returns
        -------
            X : ndarray of shape (n_x, 1)
                Mean of transformed sigma points
            P : ndarray of shape (n_x, n_x)
                Covariance of transformed sigma points

        Notes
        -----
            where:
                n_x : number of states in state vector
                n_s : number of sigma points 
                k : time step
        """
        x = np.dot(W_m, Y_sigma)
        P = np.zeros((self._n_x, self._n_x))

        for idx in range(self._n_s):
            dif = (Y_sigma[idx] - x).reshape(-1,1)
            
            P += W_c[idx] * (np.dot(dif, dif.T))

        P += Q
        return x, P
    
    def _measurement_unscented_transform(self, Z_sigma, W_m, W_c, R):
        """
        Performs an unscented transform on predicted measurement
        sigma points which have been through measurement model.

        Parameters
        ----------
            Z_sigma : ndarray of shape (n_s, n_m)
                Sigma points passed through measurement model
            W_m : ndarray of shape (n_s, 1)
                Mean Weights
            W_c : ndarray of shape (n_s, 1)
                Covariance Weights
            R : ndarray of shape (n_m, n_m)
                Measurement noise

        Returns
        -------
            Z : ndarray of shape (n_m, 1)
                Predicted measurement mean
            P_z : ndarray of shape (n_m, n_m)

        Notes
        -----
            where:
                n_x : number of states in state vector
                n_m : number of measured states
                n_s : number of sigma points 
                k : time step
        """
        Z = np.dot(Z_sigma , W_m)
        P_z = np.zeros((self._n_m, self._n_m))
        
        for idx in range(self._n_s):
            dif = (Z_sigma[:,idx] - Z).reshape(-1,1)

            P_z += W_c[idx] * (np.dot(dif, dif.T))
        
        P_z += R
        
        return Z, P_z
    
    def _cross_covariance(self, Y_sigma, X, Z_sigma, Z, W_c):
        """
        Calculates the cross covariance of the staate and the 
        measurement. 

        Parameters
        ----------
            Y_sigma : ndarray of shape (n_s, n_x)
                Projected sigma points
            X : ndarray of shape (n_x, 1)
                Predicted state mean
            Z_sigma : ndarray of shape (n_s, n_m)
                Sigma points passed through measurement model
            W_c : ndarray of shape (n_s, 1)
                Covariance Weights
            Z : ndarray of shape (n_m, 1)
                Predicted measurement mean
        Returns
        -------
            P_xz : ndarray of shape (n_x, n_m)
                Cross covariance of x and z

        Notes
        -----
            where:
                n_x : number of states in state vector
                n_m : number of measured states
                n_s : number of sigma points 
                k : time step

        """
        P_xz = np.zeros((self._n_x, self._n_m))

        for idx in range(self._n_s):
            dif_x = (Y_sigma[idx] - X).reshape(-1,1)
            dif_z = (Z_sigma[:,idx] - Z).reshape(-1,1)

            P_xz += W_c[idx] * (np.dot(dif_x, dif_z.T))
            
        return P_xz 
    
    def _h_constant_acceleration(self, Y_sigma):
        """
        Measurement model will only measure the x and y coordinate of 
        the state.
        
        Parameters
        ----------
            Y_sigma : ndarray of shape (n_s, n_x)
                Projected Sigma points 
        Returns
        -------
            Z_sigma : ndarray of shape (n_s, n_m)
        Notes
        -----
            where:
                n_x : number of states in state vector
                n_m : number of measured states
                n_s : number of sigma points 
                k : time step
        """
        Z_sigma = np.zeros((self._n_m, self._n_s))
        H = np.zeros((self._n_m, self._n_x))

        for idx in range(self._n_m):
            H[idx, idx] = 1

        Z_sigma = np.dot(H, Y_sigma.T)
        return Z_sigma

    def _measurement_update(self, Y_sigma):
        """
        Passes the propagated sigma points through the measurement model,
        this model will only track the x and y coordinates of the state.
        
        Parameters
        ----------
            Y_sigma : ndarray of shape (n_s, n_x)
                Projected Sigma points 

        Returns
        -------
            Z_sigma : ndarray of shape (n_s, n_m)
                Predicted transformed sigma points through measurement
                model

        Notes
        -----
            where:
                n_x : number of states in state vector
                n_m : number of measured states 
                n_s : number of sigma points 
                k : time step
        """
        return self._h_constant_acceleration(Y_sigma)
    
    def _residual_measurement(self, z, Z_mean):
        """
        Computes the residual of the measurement.

        Parameters
        ----------
            z : ndarray of shape (n_m, 1)
                Measured state
            Z_mean : ndarray of shape (n_m, 1)
                Mean of measured states
        Returns
        -------
            y : ndarray of shape (n_m, 1)
                Innovation
        Notes
        -----
            where,
                n_m : number of measured states 
        
        """
        return z - Z_mean
    
    def _kalman_gain(self, P_xz, P_z):
        """
        Computes the Kalman Gain, which is the difference in belief
        in state over the belief in measurement

        Parameters
        ----------
            P_xz : ndarray of shape (n_x, n_m)
                Cross covariance of x and z
            P_z : ndarray of shape (n_m, n_m)
                Predicted measurement covariance
            
        Returns
        -------
            K : ndarray of shape (n_x, n_m)
                Kalman Gain
        Notes
        -----
            where,
                n_x : number of states in vector
                n_m : number of measured states 
                n_s : number of sigma points 
        """

        K = np.dot(P_xz, np.linalg.inv(P_z))
        return K
    
    def _state_update(self, X, K, y):
        """
        Updates the predicted state. 

        Parameters
        ----------
            X : ndarray of shape (n_x, 1)
                Predicted state
            K : ndarray of shape (n_x, n_m)
                Kalman Gain
            y : ndarray of shape (n_m, 1)
                Residual

        Returns
        -------
            x : ndarray of shape (n_x, n_m)
                Updated state vector
        Notes
        -----
            where,
                n_x : number of states in vector
                n_m : number of measured states 
        """
        x = X + (np.dot(K, y))
        return x

    def _covariance_update(self, P_pred, K, P_z):
        """
        Updates the covariance after update

        Parameters
        ----------
            P_pred : ndarray of shape (n_x, n_x)
                Covariance of transformed sigma points
            K : ndarray of shape (n_x, n_m)
                Kalman Gain
            P_z : ndarray of shape (n_x, n_x)
                Covariance of measurement sigma points

        Returns
        -------
            P : ndarray of shape (n_x, n_x)
                Updated covariance of state
        
        Notes
        -----
            where,
                n_x : number of states in vector
                n_m : number of measured states 
                
        """
        P = P_pred - (K @ P_z @ K.T)
        return P
    
    def predict(self, x, P):
        """
        Runs Kalman filter prediction step.

        Parameters
        ----------
        x : ndarray of shape (n_x, 1)
            Estimated system state vector at time k
        P : ndarray of shape (n_x, n_x)
            Current uncertainty covariance matrix at time k

        Returns
        -------
        x_pred : ndarray of shape (n_x, 1)
            Estimated system state vector at time k+1
        P_pred : ndarray of shape (n_x, n_x)
            Estimated uncertainty covariance matrix at time k+1
        X_sigma : ndarray of shape (n_s, n_x)
            Generated sigma points
        W_c : ndarray of shape (n_s, 1)
            Weights for covariance
        W_m : ndarray of shapoe (n_s, 1)
            Weights for mean

        Notes
        -----
        where:
            n_x : number of states in state vector
            n_s : number of sigma points 
            k : time step
        """
        std_pos = [ 
            2 * self._std_weight_pos * x[3],
            2 * self._std_weight_pos * x[3],
            1e-2,
            2 * self._std_weight_pos * x[3],
        ]

        std_vel = [ 
            10 * self._std_weight_vel * x[3],
            10 * self._std_weight_vel * x[3],
            1e-5,
            10 * self._std_weight_vel * x[3],
        ]

        std_acc = [ 
            10 * self._std_weight_acc * x[3],
            10 * self._std_weight_acc * x[3],
            1e-5,
            10 * self._std_weight_acc * x[3],
        ]
        
        Q = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))
        X_sigma, W_m, W_c = self._scaled_sigma_point_generation(x, P)
        Y_sigma = self._project_sigma_points(X_sigma)
        x_pred, P_pred = self._unscented_transform(Y_sigma, W_c, W_m, Q)
        
        return x_pred, P_pred, Y_sigma, W_m, W_c

    def update(self, x, P, Y_sigma, W_m, W_c, z):
        """
        Runs Kalman filters update step.
        
        Parameters
        ----------
        x : ndarray of shape (n_x, 1)
            Estimated system state vector at time k
        P : ndarray of shape (n_x, n_x)
            Estimated uncertainty covariance matrix at time k
        X_sigma : ndarray of shape (n_s, n_x)
            Generated sigma points
        W_c : ndarray of shape (n_s, 1)
            Weights for covariance
        W_m : ndarray of shapoe (n_s, 1)
            Weights for mean
        z : ndarray of shape (n_m, 1)
            Measured bounding box location (x,y,a,h) at time k

        Returns
        -------
        x_upd : ndarray of shape (n_x, 1)
            Updated system state vector for time k
        P_upd : ndarray of shape (n_x, n_x)
            Updated Estimated uncertainty covariance matrix for time k
        
        Notes
        -----
        where:
            n_x : number of states in state vector
            n_s : number of sigma points 
            k : time step
        """
        R = np.diag(np.square([self._std_weight_pos * x[3]] * 4))
        Z_sigma = self._measurement_update(Y_sigma)
        z_pred, P_z = self._measurement_unscented_transform(Z_sigma, W_m, W_c, R)
        P_xz = self._cross_covariance(Y_sigma, x, Z_sigma, z_pred, W_c)
        K = self._kalman_gain(P_xz, P_z)
        y = self._residual_measurement(z, z_pred)
        x_upd = self._state_update(x, K, y)
        P_upd = self._covariance_update(P, K, P_z)

        return x_upd, P_upd
    
    # TODO : Implement if I have time
    def multi_predict(self, x, P):
        pass

    # =======================================================
    #                   OLD FUNCTIONS
    # =======================================================
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

    def compute_iou(self, pred_bbox, bboxes):
        """
        Compute the IoU between the bbox and the bboxes
        """
        ious = []
        pred_bbox = self.xyah_to_xyxy(pred_bbox)
        for bbox in bboxes:
            iou = self._compute_iou(pred_bbox, bbox)
            ious.append(iou)
        return ious

    def _compute_iou(self, bbox1, bbox2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bbox1 : list
            The first bounding box in the format [x1, y1, x2, y2].
        bbox2 : list
            The second bounding box in the format [x1, y1, x2, y2].
        Returns
        -------
        float
            The IoU of the two bounding boxes.
        """
        if bbox2 == [0, 0, 0, 0]:
            return 0
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        # Calculate intersection area
        intersection_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
        # Calculate union area
        union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersection_area
        # Calculate IoU
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def xyxy_to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            h = 1
        return [xc, yc, w / h, h]

    def xyah_to_xyxy(self, bbox):
        xc, yc, a, h = bbox
        x1 = xc - a * h / 2
        y1 = yc - h / 2
        x2 = xc + a * h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]
