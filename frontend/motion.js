// Limits are stored in CAD coordinates. A home change translates both ends equally.
export function poseRange(joint, reference = {}) {
  const offset = reference[joint.id] ?? 0;
  if (['revolute', 'prismatic'].includes(joint.type)) {
    return [joint.lower - offset, joint.upper - offset];
  }
  return [-Math.PI, Math.PI];
}

export function absolutePose(project, motion = {}) {
  return Object.fromEntries(project.joints.filter(joint => joint.type !== 'fixed' && !joint.mimic)
    .map(joint => [joint.id, (project.zero_pose?.[joint.id] ?? 0) + (motion[joint.id] ?? 0)]));
}

export function relativePose(project, absolute) {
  return Object.fromEntries(Object.entries(absolute)
    .map(([key, value]) => [key, value - (project.zero_pose?.[key] ?? 0)]));
}

export function reverseJointDirection(project, joint, motion) {
  const opposite = value => -value || 0;
  joint.axis = joint.axis.map(opposite);
  [joint.lower, joint.upper] = [opposite(joint.upper), opposite(joint.lower)];
  if (Object.hasOwn(project.zero_pose, joint.id)) project.zero_pose[joint.id] = opposite(project.zero_pose[joint.id]);
  if (Object.hasOwn(motion, joint.id)) motion[joint.id] = opposite(motion[joint.id]);
  if (joint.mimic) {
    joint.mimic.multiplier = opposite(joint.mimic.multiplier);
    joint.mimic.offset = opposite(joint.mimic.offset);
  }
  for (const follower of project.joints) {
    if (follower.mimic?.joint === joint.id) follower.mimic.multiplier = opposite(follower.mimic.multiplier);
  }
}

export function jointReviewGroup(joint) {
  const names = joint.name + ' ' + joint.child;
  if (/(?:^|[\s_-])(?:head|neck)(?=[_\s-]|$)|头|颈/i.test(names)) return '头部';
  const side = /(?:^|[\s_-])(?:left|l)(?=[_\s-]|$)|左/i.test(names) ? '左' :
    /(?:^|[\s_-])(?:right|r)(?=[_\s-]|$)|右/i.test(names) ? '右' : null;
  if (side && /shoulder|elbow|wrist|arm|hand|肩|肘|腕|臂|手/i.test(names)) return side + '臂';
  if (side && /hip|knee|ankle|thigh|shin|leg|foot|髋|膝|踝|腿|脚/i.test(names)) return side + '腿';
  return '其他关节';
}
