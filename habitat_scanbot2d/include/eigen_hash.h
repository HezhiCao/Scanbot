#ifndef SCANBOT_EIGEN_HASH_H_
#define SCANBOT_EIGEN_HASH_H_

#include <Eigen/Core>

// https://github.com/ethz-asl/aslam_cv2/blob/70da8aae9a3adb6a7f887ec6b89ca564f97e7fb8/aslam_cv_common/include/aslam/common/eigen-hash.h
namespace std {

template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
  size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols> &matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < static_cast<size_t>(matrix.size()); ++i) {
        Scalar elem = *(matrix.data() + i);
        seed ^= hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

} // namespace std

#endif // SCANBOT_EIGEN_HASH_H_
