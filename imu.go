package imu

import (
	"context"
	"math"
)

// 弧度转角度的系数
const RAD_TO_DEG = 180 / math.Pi

// 四元数, w,x,y,z
type Quat [4]float64

// https://github.com/Unity-Technologies/UnityCsReference/blob/af451c32afd761852b06ced366bb04c3d5d40e16/Runtime/Export/Math/Quaternion.cs#L87
func (lhs Quat) Multiply(rhs Quat) Quat {
	x := lhs[0]*rhs[1] + lhs[1]*rhs[0] + lhs[2]*rhs[3] - lhs[3]*rhs[2]
	y := lhs[0]*rhs[2] + lhs[2]*rhs[0] + lhs[3]*rhs[1] - lhs[1]*rhs[3]
	z := lhs[0]*rhs[3] + lhs[3]*rhs[0] + lhs[1]*rhs[2] - lhs[2]*rhs[1]
	w := lhs[0]*rhs[0] - lhs[1]*rhs[1] - lhs[2]*rhs[2] - lhs[3]*rhs[3]
	return Quat{w, x, y, z}
}

// 6轴IMU状态
type IMU6Axis struct {
	//加速度，单位：g
	AccelX float64
	AccelY float64
	AccelZ float64
	//角速度，单位：rad/s
	GyroX float64
	GyroY float64
	GyroZ float64
}

// Mahony算法的客户端，用于从服务协程中获取欧拉角
type MahonyClient struct {
	imuChan   chan updateRequest
	resetChan chan Quat
}

type updateRequest struct {
	IMU6Axis
	reply chan<- Quat
}

// 使用IMU状态更新四元数
func (c MahonyClient) UpdateIMU(imu IMU6Axis) Quat {
	reply := make(chan Quat, 1)
	c.imuChan <- updateRequest{
		imu,
		reply,
	}
	return <-reply
}

type MahonyConfig struct {
	InitQuat   Quat    //初始四元数
	SampleFreq float64 // sample frequency in Hz
	TwoKpDef   float64 // 2 * proportional gain
	TwoKiDef   float64 // 2 * integral gain
}

func DefaultMahonyConfig() MahonyConfig {
	return MahonyConfig{
		InitQuat:   Quat{1.0, 0, 0, 0},
		SampleFreq: 76.0,
		TwoKpDef:   3.5,
		TwoKiDef:   0.05,
	}
}

// 启动Mahony算法服务协程
func StartMahonyServer(ctx context.Context, config MahonyConfig) MahonyClient {
	if config == (MahonyConfig{}) {
		config = DefaultMahonyConfig()
	}
	imuChan := make(chan updateRequest)
	resetChan := make(chan Quat)
	client := MahonyClient{
		imuChan,
		resetChan,
	}
	go runMahony(ctx, config, client)
	return client
}

// Mahony服务
func runMahony(ctx context.Context, config MahonyConfig, client MahonyClient) {
	state := MahonyAHRS{
		MahonyConfig: config,
		q:            config.InitQuat,
	}
	for {
		select {
		case <-ctx.Done():
			return
		case quat := <-client.resetChan:
			state.q = quat
		case req := <-client.imuChan:
			state.updateIMU(req.IMU6Axis)
			req.reply <- state.q
		}
	}
}

type MahonyAHRS struct {
	MahonyConfig
	// integral error terms scaled by Ki
	integralFBx float64
	integralFBy float64
	integralFBz float64
	q           Quat
}

// ---------------------------------------------------------------------------------------------------
// IMU algorithm update
func (state *MahonyAHRS) updateIMU(imu IMU6Axis) {
	var recipNorm float64
	var halfvx, halfvy, halfvz float64
	var halfex, halfey, halfez float64
	var qa, qb, qc float64

	gx, gy, gz, ax, ay, az := imu.GyroX, imu.GyroY, imu.GyroZ, imu.AccelX, imu.AccelY, imu.AccelZ
	q0, q1, q2, q3 := state.q[0], state.q[1], state.q[2], state.q[3]
	twoKi := state.TwoKiDef
	twoKp := state.TwoKpDef
	sampleFreq := state.SampleFreq

	// Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
	if !((ax == 0.0) && (ay == 0.0) && (az == 0.0)) {

		// Normalise accelerometer measurement
		recipNorm = invSqrt(ax*ax + ay*ay + az*az)
		ax *= recipNorm
		ay *= recipNorm
		az *= recipNorm

		// Estimated direction of gravity and vector perpendicular to magnetic flux
		halfvx = q1*q3 - q0*q2
		halfvy = q0*q1 + q2*q3
		halfvz = q0*q0 - 0.5 + q3*q3

		// Error is sum of cross product between estimated and measured direction of gravity
		halfex = (ay*halfvz - az*halfvy)
		halfey = (az*halfvx - ax*halfvz)
		halfez = (ax*halfvy - ay*halfvx)

		// Compute and apply integral feedback if enabled
		if twoKi > 0.0 {
			state.integralFBx += twoKi * halfex * (1.0 / sampleFreq) // integral error scaled by Ki
			state.integralFBy += twoKi * halfey * (1.0 / sampleFreq)
			state.integralFBz += twoKi * halfez * (1.0 / sampleFreq)
			gx += state.integralFBx // apply integral feedback
			gy += state.integralFBy
			gz += state.integralFBz
		} else {
			state.integralFBx = 0.0 // prevent integral windup
			state.integralFBy = 0.0
			state.integralFBz = 0.0
		}

		// Apply proportional feedback
		gx += twoKp * halfex
		gy += twoKp * halfey
		gz += twoKp * halfez
	}

	// Integrate rate of change of quaternion
	gx *= (0.5 * (1.0 / sampleFreq)) // pre-multiply common factors
	gy *= (0.5 * (1.0 / sampleFreq))
	gz *= (0.5 * (1.0 / sampleFreq))
	qa = q0
	qb = q1
	qc = q2
	q0 += (-qb*gx - qc*gy - q3*gz)
	q1 += (qa*gx + qc*gz - q3*gy)
	q2 += (qa*gy - qb*gz + q3*gx)
	q3 += (qa*gz + qb*gy - qc*gx)

	// Normalise quaternion
	recipNorm = invSqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
	state.q[0] = q0 * recipNorm
	state.q[1] = q1 * recipNorm
	state.q[2] = q2 * recipNorm
	state.q[3] = q3 * recipNorm

}

// ---------------------------------------------------------------------------------------------------
// Fast inverse square-root
// See: http://en.wikipedia.org/wiki/Fast_inverse_square_root
func invSqrt(x float64) float64 {
	// halfx := 0.5 * x
	// y := x
	// buf := new(bytes.Buffer)
	// binary.Write(buf, binary.BigEndian, y)
	// // fmt.Printf("% x\n", buf.Bytes())
	// var i int32
	// binary.Read(buf, binary.BigEndian, &i)
	// // println(i)
	// i = 0x5f3759df - (i >> 1)
	// binary.Write(buf, binary.BigEndian, i)
	// binary.Read(buf, binary.BigEndian, &y)
	// y = y * (1.5 - (halfx * y * y))
	// return y
	return 1 / math.Sqrt(x)
}

/*
输入：x,y,z,w　为四元数
输出：roll，pitch，yaw欧拉角，弧度
*
*/
func (q Quat) ToEulerAngle() (roll, pitch, yaw float64) {
	w, x, y, z := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	// roll (x-axis rotation)
	sinr_cosp := +2.0 * (w*x + y*z)
	cosr_cosp := +1.0 - 2.0*(x*x+y*y)
	roll = math.Atan2(sinr_cosp, cosr_cosp)
	// pitch (y-axis rotation)
	sinp := +2.0 * (w*y - z*x)
	if math.Abs(sinp) >= 1 {
		pitch = math.Copysign(3.1415926/2, sinp) // use 90 degrees if out of range
	} else {
		pitch = math.Asin(sinp)
	}
	// yaw (z-axis rotation)
	siny_cosp := +2.0 * (w*z + x*y)
	cosy_cosp := +1.0 - 2.0*(y*y+z*z)
	yaw = math.Atan2(siny_cosp, cosy_cosp)
	return
}

// 欧拉角转四元数
func FromEulerAngle(roll, pitch, yaw float64) Quat {
	cr2 := math.Cos(roll * 0.5)
	cp2 := math.Cos(pitch * 0.5)
	cy2 := math.Cos(yaw * 0.5)
	sr2 := math.Sin(roll * 0.5)
	sp2 := math.Sin(pitch * 0.5)
	sy2 := math.Sin(yaw * 0.5)

	w := cr2*cp2*cy2 + sr2*sp2*sy2
	x := sr2*cp2*cy2 - cr2*sp2*sy2
	y := cr2*sp2*cy2 + sr2*cp2*sy2
	z := cr2*cp2*sy2 - sr2*sp2*cy2
	return Quat{float64(w), float64(x), float64(y), float64(z)}
}
