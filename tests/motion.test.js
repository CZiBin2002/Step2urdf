import { test } from 'node:test';
import assert from 'node:assert/strict';
import { absolutePose, jointReviewGroup, relativePose, poseRange, reverseJointDirection } from '../frontend/motion.js';

test('repeated home changes preserve CAD endpoints and travel', () => {
  const joint = { id: 'j', type: 'revolute', lower: -2.9, upper: 2.9 };
  const project = { joints: [joint], zero_pose: {} };
  for (const zero of [0, 1.1, -2.3, 2.9, -2.9]) {
    project.zero_pose = { j: zero };
    const [lower, upper] = poseRange(joint, project.zero_pose);
    assert.ok(Math.abs(upper - lower - 5.8) < 1e-12);
    assert.ok(Math.abs(absolutePose(project, { j: lower }).j + 2.9) < 1e-12);
    assert.ok(Math.abs(absolutePose(project, { j: upper }).j - 2.9) < 1e-12);
    assert.deepEqual(relativePose(project, absolutePose(project, { j: 0.25 })), { j: 0.25 });
  }
});

test('prismatic limits translate without silently widening or clamping invalid limits', () => {
  assert.deepEqual(poseRange({ id: 'j', type: 'prismatic', lower: 0.1, upper: 0.3 }, { j: 0.2 }),
    [-0.1, 0.3 - 0.2]);
  assert.deepEqual(poseRange({ id: 'j', type: 'revolute', lower: 1, upper: 2 }), [1, 2]);
});

test('reversing the axis preserves home, travel and coupled motion', () => {
  const joint = { id: 'motor', type: 'revolute', axis: [0, 0, 1], lower: -0.8, upper: 2.1 };
  const follower = { id: 'coupled', type: 'revolute', mimic: { joint: joint.id, multiplier: 2, offset: 0.1 } };
  const project = { joints: [joint, follower], zero_pose: { motor: 0.4 } };
  const motion = { motor: -0.13 };
  const initial = absolutePose(project, motion).motor;

  reverseJointDirection(project, joint, motion);

  assert.deepEqual(joint.axis, [0, 0, -1]);
  assert.deepEqual([joint.lower, joint.upper], [-2.1, 0.8]);
  assert.equal(project.zero_pose.motor, -0.4);
  const reversed = absolutePose(project, motion).motor;
  assert.equal(reversed, -initial);
  assert.equal(reversed * follower.mimic.multiplier + follower.mimic.offset, initial * 2 + 0.1);
});

test('OP3 limb names and generic robot joints get useful review groups', () => {
  assert.equal(jointReviewGroup({ name: 'head_yaw', child: 'neck' }), '头部');
  assert.equal(jointReviewGroup({ name: 'left_shoulder_pitch', child: 'left_shoulder' }), '左臂');
  assert.equal(jointReviewGroup({ name: '右膝', child: '小腿' }), '右腿');
  assert.equal(jointReviewGroup({ name: 'base_yaw', child: 'turret' }), '其他关节');
});
