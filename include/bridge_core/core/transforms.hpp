#ifndef BRIDGE_CORE_TRANSFORMS_HPP
#define BRIDGE_CORE_TRANSFORMS_HPP

#include <array>
#include <cmath>

namespace bridge_core {

/**
 * @brief Get conjugate of a quaternion
 * @param quat Quaternion [w, x, y, z]
 * @return Conjugate quaternion [w, -x, -y, -z]
 */
inline std::array<float, 4> quatConjugate(const std::array<float, 4>& quat) {
    return {quat[0], -quat[1], -quat[2], -quat[3]};
}

/**
 * @brief Multiply two quaternions
 * @param q1 First quaternion [w, x, y, z]
 * @param q2 Second quaternion [w, x, y, z]
 * @return Product q1 * q2
 */
inline std::array<float, 4> quatMultiply(const std::array<float, 4>& q1, 
                                         const std::array<float, 4>& q2) {
    return {
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3], // w
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2], // x
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1], // y
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]  // z
    };
}

/**
 * @brief Rotate vector by inverse of quaternion (q^-1 * v * q)
 * @param quat Quaternion [w, x, y, z]
 * @param vec Vector [x, y, z]
 * @return Rotated vector
 */
inline std::array<float, 3> quatRotateInv(const std::array<float, 4>& quat, 
                                          const std::array<float, 3>& vec) {
    // Convert vector to pure quaternion (0, x, y, z)
    std::array<float, 4> v_quat = {0.0f, vec[0], vec[1], vec[2]};
    
    // Get quaternion conjugate (inverse for unit quaternion)
    std::array<float, 4> q_inv = quatConjugate(quat);
    
    // Quaternion rotation: result = q_inv * v * q
    std::array<float, 4> tmp = quatMultiply(q_inv, v_quat);
    std::array<float, 4> result = quatMultiply(tmp, quat);
    
    return {result[1], result[2], result[3]};
}

/**
 * @brief Rotate vector by quaternion (q * v * q^-1)
 * @param quat Quaternion [w, x, y, z]
 * @param vec Vector [x, y, z]
 * @return Rotated vector
 */
inline std::array<float, 3> quatRotate(const std::array<float, 4>& quat, 
                                       const std::array<float, 3>& vec) {
    // Convert vector to pure quaternion (0, x, y, z)
    std::array<float, 4> v_quat = {0.0f, vec[0], vec[1], vec[2]};
    
    // Get quaternion conjugate (inverse for unit quaternion)
    std::array<float, 4> q_inv = quatConjugate(quat);
    
    // Quaternion rotation: result = q * v * q_inv
    std::array<float, 4> tmp = quatMultiply(quat, v_quat);
    std::array<float, 4> result = quatMultiply(tmp, q_inv);
    
    return {result[1], result[2], result[3]};
}

/**
 * @brief Convert quaternion to Euler angles (XYZ convention: roll, pitch, yaw)
 * @param quat Quaternion [w, x, y, z]
 * @return Euler angles [roll, pitch, yaw] in radians
 */
inline std::array<float, 3> quatToEuler(const std::array<float, 4>& quat) {
    const float w = quat[0];
    const float x = quat[1];
    const float y = quat[2];
    const float z = quat[3];

    // Roll (X-axis rotation)
    const float sinr_cosp = 2.0f * (w * x + y * z);
    const float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    const float roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (Y-axis rotation)
    const float sinp = 2.0f * (w * y - z * x);
    float pitch;
    if (std::abs(sinp) >= 1.0f) {
        // Handle gimbal lock
        pitch = std::copysign(static_cast<float>(M_PI) / 2.0f, sinp);
    } else {
        pitch = std::asin(sinp);
    }

    // Yaw (Z-axis rotation)
    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
    const float yaw = std::atan2(siny_cosp, cosy_cosp);

    return {roll, pitch, yaw};
}

/**
 * @brief Convert Euler angles to quaternion (XYZ convention)
 * @param euler Euler angles [roll, pitch, yaw] in radians
 * @return Quaternion [w, x, y, z]
 */
inline std::array<float, 4> eulerToQuat(const std::array<float, 3>& euler) {
    const float roll = euler[0] * 0.5f;
    const float pitch = euler[1] * 0.5f;
    const float yaw = euler[2] * 0.5f;
    
    const float cr = std::cos(roll);
    const float sr = std::sin(roll);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    
    return {
        cr * cp * cy + sr * sp * sy, // w
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy  // z
    };
}

/**
 * @brief Normalize a quaternion
 * @param quat Quaternion [w, x, y, z]
 * @return Normalized quaternion
 */
inline std::array<float, 4> quatNormalize(const std::array<float, 4>& quat) {
    float norm = std::sqrt(quat[0]*quat[0] + quat[1]*quat[1] + 
                          quat[2]*quat[2] + quat[3]*quat[3]);
    if (norm < 1e-8f) {
        return {1.0f, 0.0f, 0.0f, 0.0f}; // Return identity if near zero
    }
    return {quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm};
}

/**
 * @brief Clamp a value between min and max
 */
template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

/**
 * @brief Linear interpolation
 */
template<typename T>
inline T lerp(T a, T b, float t) {
    return a + t * (b - a);
}

} // namespace bridge_core

#endif // BRIDGE_CORE_TRANSFORMS_HPP

