import {
  ArrowDownToLine, ArrowRight, Box, Check, ChevronDown, CircleHelp, CloudUpload,
  Download, Eye, EyeOff, FileCode, Folder, FolderOpen, GitBranch, Layers3, Link2,
  Maximize, Mouse, Plus, Redo2, RotateCcw, Save, ScanLine, Search, SlidersHorizontal,
  Sparkles, Trash2, Undo2, X, createIcons,
} from 'lucide';

import './style.css';
import { Viewer, jointCoordinate } from './viewer.js';

const $ = selector => document.querySelector(selector);
const icon = name => '<i data-lucide="' + name + '"></i>';
const icons = {
  ArrowDownToLine, ArrowRight, Box, Check, ChevronDown, CircleHelp, CloudUpload,
  Download, Eye, EyeOff, FileCode, Folder, FolderOpen, GitBranch, Layers3, Link2,
  Maximize, Mouse, Plus, Redo2, RotateCcw, Save, ScanLine, Search, SlidersHorizontal,
  Sparkles, Trash2, Undo2, X,
};
const paintIcons = () => createIcons({ icons, attrs: { 'stroke-width': 1.65 } });
const esc = value => String(value ?? '').replace(/[&<>"']/g, character =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
const clone = value => structuredClone(value);
const id = () => crypto.randomUUID().slice(0, 12);
const zero = [0, 0, 0];
const unitZ = [0, 0, 1];
const kindLabel = { fixed: '固定', revolute: '旋转', continuous: '连续旋转', prismatic: '滑动', ball: '球铰' };

let job;
let project;
let parts = {};
let selection = { kind: 'links', id: null };
let treeTab = 'links';
let inspector = 'properties';
let step = 'configure';
let chosenParts = new Set();
let hidden = new Set();
let hiddenLinks = new Set();
let expanded = new Set();
let folded = new Set();
let onlyUnassigned = false;
let pose = {};
let collisions = false;
let undo = [];
let redo = [];
let aiResult;
let axisOptions = [];
let loading = false;
let latestFiles = [];
let preferredModel = '';
let massProperties = {};
let pendingMass = 0;
let toastTimeout;
let saveRequest = Promise.resolve();
let exportOptions = { formats: ['urdf'], merge: false, training: 'isaac' };

const dialog = $('#dialog');
let viewer;
try {
  viewer = new Viewer($('#canvasHost'), selectPart);
} catch {
  $('#viewMessage').textContent = '此设备无法初始化 WebGL，仍可编辑结构和导出模型。';
}

async function api(path, method = 'GET', body) {
  const headers = body && !(body instanceof FormData) ? { 'Content-Type': 'application/json' } : {};
  const response = await fetch('/api' + path, {
    method,
    headers,
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const message = Array.isArray(error.detail)
      ? error.detail.map(item => item.loc.slice(1).join('.') + '：' + item.msg).join('；')
      : error.detail;
    throw new Error(message || '请求失败 (' + response.status + ')');
  }
  return response;
}

const projectApi = (operation, data = project, method = 'POST') => api(
  '/projects/' + job.id + (operation ? '/' + operation : ''), method, data,
);

function toast(text, error = false) {
  const target = $('#toast');
  clearTimeout(toastTimeout);
  target.classList.toggle('error', error);
  target.textContent = text;
  target.hidden = false;
  toastTimeout = setTimeout(() => { target.hidden = true; }, error ? 7000 : 3200);
}

function cacheProject(reportError = true) {
  if (!job) return;
  const path = '/projects/' + job.id;
  const data = clone(project);
  saveRequest = saveRequest.catch(() => {}).then(() => api(path, 'PUT', data));
  if (reportError) {
    saveRequest.catch(error => toast('无法保存到缓存：' + error.message, true));
  }
  return saveRequest;
}

function edit(mutate) {
  undo.push(clone(project));
  if (undo.length > 100) undo.shift();
  redo = [];
  mutate();
  const movable = new Set(project.joints.filter(joint => joint.type !== 'fixed' && !joint.mimic).map(joint => joint.id));
  for (const key of Object.keys(project.zero_pose)) {
    if (!movable.has(key)) delete project.zero_pose[key];
  }
  for (const key of Object.keys(pose)) {
    if (!movable.has(key)) delete pose[key];
  }
  massProperties = {};
  aiResult = null;
  axisOptions = [];
  render();
  cacheProject();
}

function history(undoing) {
  const source = undoing ? undo : redo;
  const target = undoing ? redo : undo;
  if (!source.length) return;
  target.push(clone(project));
  project = source.pop();
  pose = {};
  massProperties = {};
  aiResult = null;
  render();
  cacheProject();
}

function load(data) {
  job = data;
  project = data.project;
  project.zero_pose ??= {};
  parts = data.parts;
  selection = { kind: 'links', id: project.links[0]?.id ?? null };
  treeTab = project.links.length ? 'links' : 'parts';
  inspector = 'properties';
  step = 'configure';
  chosenParts = new Set();
  hidden = new Set();
  hiddenLinks = new Set();
  expanded = new Set();
  folded = new Set();
  onlyUnassigned = false;
  pose = {};
  undo = [];
  redo = [];
  aiResult = null;
  axisOptions = [];
  massProperties = {};
  localStorage.setItem('step2urdf.last', job.id);
  viewer?.load(parts);
  $('#viewMessage').hidden = Boolean(viewer);
  if (dialog.open) dialog.close();
  render();
  viewer?.fit();
}

function syncView() {
  viewer?.sync(project, {
    kind: selection.kind, id: selection.id, parts: chosenParts, hidden, hiddenLinks, pose, collisions, highlight: true,
  });
}

function render() {
  if (!project) return;
  $('#projectName').textContent = project.name;
  $('#counts').textContent = Object.keys(parts).length + ' 零件  /  ' + project.links.length + ' Link  /  ' + project.joints.length + ' 关节';
  $('#sourceLabel').textContent = job.source;
  $('#undo').disabled = !undo.length;
  $('#redo').disabled = !redo.length;
  for (const button of document.querySelectorAll('[data-step]')) button.classList.toggle('active', button.dataset.step === step);
  renderTree();
  renderInspector();
  syncView();
}

function treeDepth(id) {
  const seen = new Set();
  let depth = 0;
  let incoming = project.joints.find(joint => joint.child === id);
  while (incoming && !seen.has(incoming.parent)) {
    seen.add(incoming.parent);
    depth++;
    incoming = project.joints.find(joint => joint.child === incoming.parent);
  }
  return Math.min(depth, 6);
}

function renderTree() {
  if (!project) return;
  for (const button of document.querySelectorAll('[data-tree]')) button.classList.toggle('active', button.dataset.tree === treeTab);
  const query = $('#search').value.trim().toLocaleLowerCase();
  const entries = treeTab === 'parts' ? Object.values(parts) : project[treeTab];
  const tree = $('#tree');
  const assignment = new Set(project.links.flatMap(link => link.parts));
  const matches = entries.filter(entry =>
    (treeTab !== 'parts' || !onlyUnassigned || !assignment.has(entry.id)) &&
    (entry.name + ' ' + (entry.path || []).join(' ')).toLocaleLowerCase().includes(query));
  const filter = treeTab === 'parts' ? '<label class="check-row parts-filter"><input type="checkbox" id="unassigned"' +
    (onlyUnassigned ? ' checked' : '') + '>仅显示未分配零件</label>' : '';
  if (!matches.length) {
    tree.innerHTML = filter + '<div class="empty-tree">' + (query ? '没有匹配项' : '当前列表为空') + '</div>';
  } else {
    const renderRow = entry => {
      const key = esc(entry.id ?? entry.name);
      const isPart = treeTab === 'parts';
      const isLink = treeTab === 'links';
      const selected = (selection.kind === treeTab && selection.id === (entry.id ?? entry.name)) ||
        (isPart && chosenParts.has(entry.id));
      const depth = isLink ? treeDepth(entry.id) : 0;
      const visibility = isPart ? hidden.has(entry.id) : hiddenLinks.has(entry.id);
      const label = isPart ? (assignment.has(entry.id) ? '已分配' : '未分配') :
        isLink ? entry.parts.length + ' 件' : kindLabel[entry.type] || (entry.type === 'weld' ? '焊接' : '连接');
      const symbol = treeTab === 'joints' ? 'git-branch' : treeTab === 'loops' ? 'link-2' : 'box';
      const expand = isLink && entry.parts.length
        ? '<button class="twisty icon-button" data-expand="' + key + '" aria-label="展开或收起 Link 零件" aria-expanded="' +
          expanded.has(entry.id) + '">' + icon('chevron-down') + '</button>'
        : '<span class="twisty-space"></span>';
      const row = '<div class="tree-row ' + (selected ? 'selected ' : '') + (visibility ? 'muted' : '') +
        '" style="--depth:' + depth + '">' + expand + '<button class="tree-select" data-select="' + key + '"><span class="tree-icon ' +
        (treeTab === 'joints' ? 'joint' : '') + '">' + icon(symbol) + '</span><span class="tree-name">' +
        esc(entry.name) + '</span><span class="tree-tag">' + esc(label) + '</span></button>' +
        (isPart || isLink ? '<button class="icon-button visibility" data-visibility="' + key + '" aria-label="' +
          (visibility ? '显示' : '隐藏') + esc(entry.name) + '">' + icon(visibility ? 'eye-off' : 'eye') + '</button>' : '') + '</div>';
      if (!isLink || !expanded.has(entry.id)) return row;
      return row + entry.parts.map(partId => {
        const part = parts[partId];
        if (!part) return '';
        return '<div class="tree-row leaf ' + (chosenParts.has(partId) ? 'selected' : '') + '" style="--depth:' +
          (depth + 1) + '"><button class="tree-select" data-part="' + esc(partId) + '"><span class="tree-icon">' +
          icon('box') + '</span><span class="tree-name">' + esc(part.name) + '</span></button></div>';
      }).join('');
    };
    if (treeTab === 'parts') {
      const root = { groups: new Map(), parts: [] };
      for (const entry of matches) {
        let group = root;
        const path = entry.path.slice(0, -1);
        for (const [index, name] of path.entries()) {
          if (!group.groups.has(name)) {
            group.groups.set(name, { groups: new Map(), parts: [], key: JSON.stringify(path.slice(0, index + 1)) });
          }
          group = group.groups.get(name);
        }
        group.parts.push(entry);
      }
      function renderGroup(group) {
        const children = [...group.groups].map(([name, child]) =>
          '<details class="assembly-group" data-group="' + esc(child.key) + '"' + (query || !folded.has(child.key) ? ' open' : '') +
          '><summary>' + icon('folder') + esc(name) + '</summary>' + renderGroup(child) + '</details>').join('');
        return children + group.parts.map(renderRow).join('');
      }
      tree.innerHTML = filter + renderGroup(root);
      tree.querySelectorAll('[data-group] summary').forEach(summary => summary.onclick = () => {
        const group = summary.parentElement;
        if (group.open) folded.add(group.dataset.group); else folded.delete(group.dataset.group);
      });
    } else {
      tree.innerHTML = matches.map(renderRow).join('');
    }
  }
  $('#unassigned')?.addEventListener('change', event => { onlyUnassigned = event.currentTarget.checked; renderTree(); });
  const buttons = { links: '创建 Link', joints: '创建关节', parts: '用所选零件创建 Link', loops: '添加闭链约束' };
  const action = $('#tree-actions');
  action.innerHTML = '<button class="button secondary wide" id="addItem">' + icon('plus') + esc(buttons[treeTab]) + '</button>' +
    (treeTab === 'parts' && chosenParts.size ? '<div class="selection-actions"><span>已选 ' + chosenParts.size +
      ' 件</span><button class="text-button" id="hideSelected">隐藏所选</button><button class="text-button" id="clearSelected">清除</button></div>' : '') +
    (treeTab === 'parts' && hidden.size ? '<button class="text-button" id="restoreHidden">显示 ' + hidden.size + ' 个隐藏零件</button>' : '');
  tree.querySelectorAll('[data-select]').forEach(button => button.onclick = event => {
    if (treeTab === 'parts') return selectPart(button.dataset.select, event.shiftKey || event.ctrlKey);
    selection = { kind: treeTab, id: button.dataset.select };
    inspector = 'properties';
    axisOptions = [];
    render();
  });
  tree.querySelectorAll('[data-part]').forEach(button => button.onclick = event => selectPart(button.dataset.part, event.shiftKey || event.ctrlKey));
  tree.querySelectorAll('[data-expand]').forEach(button => button.onclick = () => {
    const key = button.dataset.expand;
    if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
    renderTree();
  });
  tree.querySelectorAll('[data-visibility]').forEach(button => button.onclick = () => {
    const set = treeTab === 'parts' ? hidden : hiddenLinks;
    const key = button.dataset.visibility;
    if (set.has(key)) set.delete(key); else set.add(key);
    renderTree();
    syncView();
  });
  $('#addItem').onclick = () => {
    if (treeTab === 'joints') return openJoint();
    if (treeTab === 'loops') return openLoop();
    openLink();
  };
  $('#hideSelected')?.addEventListener('click', () => {
    chosenParts.forEach(key => hidden.add(key));
    chosenParts.clear();
    render();
  });
  $('#clearSelected')?.addEventListener('click', () => { chosenParts.clear(); render(); });
  $('#restoreHidden')?.addEventListener('click', () => { hidden.clear(); render(); });
  paintIcons();
}

function selectPart(key, append) {
  if (!append) chosenParts.clear();
  if (chosenParts.has(key)) chosenParts.delete(key); else chosenParts.add(key);
  treeTab = 'parts';
  inspector = 'properties';
  selection = { kind: 'parts', id: key };
  render();
}

function selectedObject() {
  if (selection.kind === 'parts') return parts[selection.id];
  return project[selection.kind]?.find(entry => (entry.id ?? entry.name) === selection.id);
}

const heading = (label, name, type) => '<div class="selection-heading"><span class="selection-symbol ' + type + '">' +
  icon(type === 'joint' ? 'git-branch' : type === 'ai' ? 'sparkles' : 'box') +
  '</span><div><div class="eyebrow">' + label + '</div><h3>' + esc(name) + '</h3></div></div>';
const field = (label, key, value, type = 'text') => '<label class="field"><span>' + label +
  '</span><input data-field="' + key + '" type="' + type + '" ' + (type === 'number' ? 'step="any"' : '') +
  ' value="' + esc(value) + '"></label>';
const select = (label, key, value, entries) => '<label class="field"><span>' + label +
  '</span><select data-field="' + key + '">' + entries.map(([id, text]) => '<option value="' + esc(id) + '"' +
  (id === value ? ' selected' : '') + '>' + esc(text) + '</option>').join('') + '</select></label>';
const vectors = (label, key, value) => '<div class="vector-field"><span>' + label + '</span><div class="vector-inputs">' +
  ['X', 'Y', 'Z'].map((axis, index) => '<label><span>' + axis + '</span><input aria-label="' + label + ' ' + axis +
  '" data-vector="' + key + '" data-coordinate="' + index + '" type="number" step="any" value="' +
  Number(value[index].toPrecision(8)) + '"></label>').join('') + '</div></div>';
const linkOptions = () => project.links.map(link => [link.id, link.name]);
const stat = (label, value) => '<div><span>' + label + '</span><strong>' + value + '</strong></div>';

function renderInspector() {
  if (!project) return;
  for (const button of document.querySelectorAll('[data-inspect]')) button.classList.toggle('active', button.dataset.inspect === inspector);
  const target = $('#inspector-body');
  if (inspector === 'ai') {
    renderAI(target);
  } else if (inspector === 'pose') {
    renderPose(target);
  } else {
    const selected = selectedObject();
    if (!selected) {
      target.innerHTML = '<div class="inspector-empty">' + icon('scan-line') + '<p>选择一个零件、Link 或关节以查看属性。</p></div>';
    } else if (selection.kind === 'links') {
      renderLink(target, selected);
    } else if (selection.kind === 'joints') {
      renderJoint(target, selected);
    } else if (selection.kind === 'parts') {
      renderPart(target, selected);
    } else {
      renderLoop(target, selected);
    }
  }
  target.insertAdjacentHTML('beforeend', '<button class="button quiet wide model-settings" id="modelSettings">' +
    icon('sliders-horizontal') + '模型基准与初始位姿</button>');
  $('#modelSettings').onclick = openSettings;
  paintIcons();
}

function bindFields(target, item, shouldChange = () => true) {
  target.querySelectorAll('[data-field]').forEach(input => input.onchange = () => {
    const key = input.dataset.field;
    const value = input.type === 'number' ? input.valueAsNumber : input.value;
    if ((input.type === 'number' && !Number.isFinite(value)) || (key === 'name' && !String(value).trim()) || !shouldChange(key, value)) {
      input.value = item[key] ?? '';
      return;
    }
    edit(() => {
      if (selection.kind === 'loops' && key === 'name') selection.id = value;
      item[key] = value;
    });
  });
  target.querySelectorAll('[data-vector]').forEach(input => input.onchange = () => {
    const value = input.valueAsNumber;
    const key = input.dataset.vector;
    const index = Number(input.dataset.coordinate);
    if (!Number.isFinite(value)) { input.value = item[key][index]; return; }
    edit(() => { item[key][index] = value; });
  });
}

function renderLink(target, link) {
  const physical = massProperties[link.id];
  const incoming = project.joints.find(joint => joint.child === link.id);
  const automatic = physical ? physical.mass.toFixed(4) + ' kg' : '按密度计算';
  const inertia = physical
    ? '<div class="readout"><div><span>质心 (m)</span><strong>' + physical.center.map(x => x.toFixed(4)).join(', ') +
      '</strong></div><div><span>惯量 (kg·m²)</span><strong>' + physical.inertia.map(x => x.toExponential(2)).join(' / ') + '</strong></div></div>'
    : '';
  target.innerHTML = heading('LINK', link.name, 'link') + '<div class="stats">' +
    stat('关联零件', link.parts.length) + stat('父关节', incoming ? esc(incoming.name) : '根 Link') + '</div>' +
    '<section class="inspector-section"><h4>坐标与外观</h4>' + field('名称', 'name', link.name) +
    (incoming ? '<p class="subtle">Link 原点取自父关节的锚点。</p>' : vectors('原点 (m)', 'origin', link.origin)) +
    vectors('朝向 RPY (rad)', 'rpy', link.rpy) +
    select('碰撞体', 'collision', link.collision, [['mesh', '网格'], ['box', '包围盒'], ['none', '禁用']]) + '</section>' +
    '<section class="inspector-section"><h4>质量与惯量</h4>' + field('密度 (kg/m³)', 'density', link.density, 'number') +
    '<div class="readout"><div><span>质量</span><strong>' + (link.mass != null ? link.mass + ' kg' : automatic) +
    '</strong></div></div>' + inertia + '<div class="inline-actions"><button id="computeMass" class="button secondary">计算</button>' +
    '<button id="editMass" class="button secondary">手动设置</button></div></section>' +
    '<button class="button quiet danger wide" id="deleteSelected">' + icon('trash-2') + '删除 Link</button>';
  bindFields(target, link, (key, value) => key !== 'density' || value > 0);
  $('#computeMass').onclick = async () => {
    const request = ++pendingMass;
    const current = job.id;
    try {
      const result = await (await projectApi('properties')).json();
      if (request !== pendingMass || current !== job.id) return;
      massProperties = result;
      renderInspector();
    } catch (error) { toast(error.message, true); }
  };
  $('#editMass').onclick = () => openMass(link, physical);
  $('#deleteSelected').onclick = () => edit(() => {
    project.links = project.links.filter(entry => entry !== link);
    const removed = project.joints.filter(entry => entry.child === link.id || entry.parent === link.id).map(entry => entry.id);
    project.joints = project.joints.filter(entry => !removed.includes(entry.id));
    project.joints.forEach(entry => { if (removed.includes(entry.mimic?.joint)) entry.mimic = null; });
    project.loops = project.loops.filter(entry => entry.first !== link.id && entry.second !== link.id);
    selection.id = project.links[0]?.id ?? null;
  });
}

function allowableJoint(parent, child, id) {
  if (parent === child || !parent || !child) return false;
  const other = project.joints.filter(joint => joint.id !== id);
  if (other.some(joint => joint.child === child)) return false;
  const seen = new Set();
  let cursor = parent;
  while (cursor && !seen.has(cursor)) {
    if (cursor === child) return false;
    seen.add(cursor);
    cursor = other.find(joint => joint.child === cursor)?.parent;
  }
  return !cursor;
}

function renderJoint(target, joint) {
  const suggestions = axisOptions.map((candidate, index) => '<button class="suggestion-axis" data-axis="' + index +
    '">候选 ' + (index + 1) + ' · ' + candidate.axis.map(x => x.toFixed(2)).join(', ') + ' · ' +
    Math.round(candidate.confidence * 100) + '%</button>').join('');
  target.innerHTML = heading('JOINT', joint.name, 'joint') +
    '<section class="inspector-section"><h4>连接与运动</h4>' + field('名称', 'name', joint.name) +
    select('父 Link', 'parent', joint.parent, linkOptions()) + select('子 Link', 'child', joint.child, linkOptions()) +
    select('关节类型', 'type', joint.type, Object.entries(kindLabel)) +
    vectors('锚点 (m)', 'anchor', joint.anchor) +
    (joint.type !== 'fixed' ? vectors('轴向', 'axis', joint.axis) : '') +
    '<button class="button secondary wide" id="suggestAxis">' + icon('sparkles') + '识别转轴</button>' +
    suggestions + '</section>' +
    (joint.type === 'fixed' ? '' : '<section class="inspector-section"><h4>关节限位与动力学</h4>' +
      (['revolute', 'prismatic'].includes(joint.type) ? '<div class="field-pair">' +
        field('下限 (' + (joint.type === 'prismatic' ? 'm' : 'rad') + ')', 'lower', joint.lower, 'number') +
        field('上限', 'upper', joint.upper, 'number') + '</div>' : '') +
      '<div class="field-pair">' + field('最大力 / 扭矩', 'effort', joint.effort, 'number') +
      field('最大速度', 'velocity', joint.velocity, 'number') + '</div><div class="field-pair">' +
      field('阻尼', 'damping', joint.damping, 'number') + field('摩擦', 'friction', joint.friction, 'number') + '</div></section>') +
    (['revolute', 'continuous', 'prismatic'].includes(joint.type) ? '<section class="inspector-section"><h4>关节耦合</h4>' +
      (joint.mimic ? '<p class="subtle">跟随 ' + esc(project.joints.find(item => item.id === joint.mimic.joint)?.name) + '</p>' : '') +
      '<button id="editMimic" class="button secondary wide">设置耦合</button></section>' : '') +
    '<button class="button quiet danger wide" id="deleteSelected">' + icon('trash-2') + '删除关节</button>';
  bindFields(target, joint, (key, value) => {
    if (key === 'parent' || key === 'child') {
      if (allowableJoint(key === 'parent' ? value : joint.parent, key === 'child' ? value : joint.child, joint.id)) return true;
      toast('该连接会形成循环、重复父关节或自连接。', true);
      return false;
    }
    if (['effort', 'velocity'].includes(key)) return value > 0;
    if (['damping', 'friction'].includes(key)) return value >= 0;
    return true;
  });
  $('#suggestAxis').onclick = async () => {
    try {
      axisOptions = await (await projectApi('axes', { project, parent: joint.parent, child: joint.child })).json();
      if (!axisOptions.length) toast('未发现同轴圆柱面，请手动设置轴向。');
      renderInspector();
    } catch (error) { toast(error.message, true); }
  };
  $('#editMimic')?.addEventListener('click', () => openMimic(joint));
  target.querySelectorAll('[data-axis]').forEach(button => button.onclick = () => {
    const candidate = axisOptions[Number(button.dataset.axis)];
    edit(() => { joint.anchor = candidate.anchor; joint.axis = candidate.axis; });
  });
  $('#deleteSelected').onclick = () => edit(() => {
    project.joints = project.joints.filter(entry => entry !== joint);
    project.joints.forEach(entry => { if (entry.mimic?.joint === joint.id) entry.mimic = null; });
    delete pose[joint.id];
    selection.id = project.joints[0]?.id ?? null;
  });
}

function renderPart(target, part) {
  const link = project.links.find(item => item.parts.includes(part.id));
  target.innerHTML = heading('CAD 零件', part.name, 'part') + '<div class="stats">' +
    stat('体积 (cm³)', (part.volume * 1e6).toFixed(2)) + stat('圆柱面', part.cylinders.length) + '</div>' +
    '<section class="inspector-section"><h4>装配位置</h4><p class="subtle">' + esc(part.path.join(' / ')) +
    '</p><div class="readout"><div><span>质心 (m)</span><strong>' + part.center.map(value => value.toFixed(4)).join(', ') +
    '</strong></div></div></section><section class="inspector-section"><h4>分配到 Link</h4>' +
    select('所属 Link', 'partLink', link?.id ?? '', [['', '未分配'], ...linkOptions()]) +
    '<p class="subtle">按住 Shift 或 Ctrl 可在零件列表和 3D 视图中多选。</p></section>';
  target.querySelector('[data-field="partLink"]').onchange = event => {
    const owner = event.currentTarget.value;
    const ids = chosenParts.has(part.id) ? [...chosenParts] : [part.id];
    edit(() => {
      for (const link of project.links) {
        link.parts = link.parts.filter(key => !ids.includes(key));
        if (link.id === owner) link.parts.push(...ids);
      }
    });
  };
}

function renderLoop(target, loop) {
  target.innerHTML = heading('闭链约束', loop.name, 'joint') + '<section class="inspector-section"><h4>约束设置</h4>' +
    field('名称', 'name', loop.name) + select('第一个 Link', 'first', loop.first, linkOptions()) +
    select('第二个 Link', 'second', loop.second, linkOptions()) +
    select('类型', 'type', loop.type, [['connect', '球形连接'], ['weld', '刚性焊接']]) +
    vectors('锚点 (m)', 'anchor', loop.anchor) +
    '<p class="subtle">闭链约束可导出为 MJCF 或 USD。</p></section>' +
    '<button class="button quiet danger wide" id="deleteSelected">' + icon('trash-2') + '删除约束</button>';
  bindFields(target, loop, (key, value) => (key !== 'first' || value !== loop.second) && (key !== 'second' || value !== loop.first));
  $('#deleteSelected').onclick = () => edit(() => {
    project.loops = project.loops.filter(entry => entry !== loop);
    selection.id = project.loops[0]?.name ?? null;
  });
}

function renderPose(target) {
  const joints = project.joints.filter(joint => joint.type !== 'fixed' && !joint.mimic);
  const customized = Object.keys(project.zero_pose).length > 0;
  target.innerHTML = heading('运动预览', '关节试转', 'joint') +
    '<p class="subtle">试转数值相对于当前零位。将姿态设为零位后，导出的模型也会以该姿态为 0。</p>' +
    (joints.length ? '<div class="zero-controls"><div class="zero-label">当前基准：' + (customized ? '自定义零位' : 'CAD 零位') +
      '</div><button class="button primary wide" id="saveZeroPose">' + icon('check') + '将当前姿态设为零位</button>' +
      '<div class="inline-actions"><button class="button secondary" id="zeroPose">' + icon('rotate-ccw') + '回到零位</button>' +
      (customized ? '<button class="text-button" id="clearZeroPose">恢复 CAD 零位</button>' : '') + '</div></div>' : '') +
    (joints.length ? '<div class="pose-list">' + joints.map(joint => {
      const zero = project.zero_pose[joint.id] ?? 0;
      const bounded = joint.type === 'revolute' || joint.type === 'prismatic';
      const lower = bounded ? Math.min(joint.lower - zero, 0) : -Math.PI;
      const upper = bounded ? Math.max(joint.upper - zero, 0) : Math.PI;
      const value = Math.max(lower, Math.min(upper, pose[joint.id] ?? 0));
      const limits = ' min="' + lower + '" max="' + upper + '" step="any"';
      const unit = joint.type === 'prismatic' ? 'm' : 'rad';
      return '<div class="pose-row"><span><strong>' + esc(joint.name) + '</strong><label class="pose-number">' +
        '<input aria-label="' + esc(joint.name) + ' 数值" type="number" data-pose-number' + limits +
        ' value="' + Number(value.toFixed(6)) + '"><span>' + unit + '</span></label></span><input aria-label="' +
        esc(joint.name) + '" type="range"' + limits + ' value="' + value + '" data-pose="' + esc(joint.id) + '"></div>';
    }).join('') + '</div>' :
    '<div class="empty-tree">配置运动关节后即可试转。</div>');
  target.querySelectorAll('[data-pose]').forEach(slider => {
    const number = slider.parentNode.querySelector('[data-pose-number]');
    const show = () => {
      pose[slider.dataset.pose] = slider.valueAsNumber;
      number.value = Number(slider.valueAsNumber.toFixed(6));
      selection = { kind: 'joints', id: slider.dataset.pose };
      syncView();
    };
    slider.oninput = show;
    number.onchange = () => {
      if (Number.isFinite(number.valueAsNumber)) slider.value = number.value;
      show();
    };
  });
  $('#zeroPose')?.addEventListener('click', resetPose);
  $('#saveZeroPose')?.addEventListener('click', async event => {
    const reference = Object.fromEntries(joints.map(joint =>
      [joint.id, (project.zero_pose[joint.id] ?? 0) + (pose[joint.id] ?? 0)]));
    const byId = new Map(project.joints.map(joint => [joint.id, joint]));
    for (const joint of project.joints) {
      if (joint.type !== 'revolute' && joint.type !== 'prismatic') continue;
      const q = jointCoordinate(joint, byId, reference);
      if (q < joint.lower - 1e-9 || q > joint.upper + 1e-9) {
        toast(joint.name + ' 的当前姿态超出关节限位。', true);
        return;
      }
    }
    if (project.loops.length) {
      const current = project;
      const button = event.currentTarget;
      button.disabled = true;
      try {
        const result = await (await projectApi('validate', { project: { ...current, zero_pose: reference }, formats: [] })).json();
        if (project !== current || !button.isConnected) return;
        if (result.errors.length) throw new Error(result.errors.join(' '));
      } catch (error) {
        toast(error.message, true);
        return;
      } finally { button.disabled = false; }
    }
    edit(() => { project.zero_pose = reference; pose = {}; });
    toast('当前姿态已设为零位。');
  });
  $('#clearZeroPose')?.addEventListener('click', () => {
    edit(() => { project.zero_pose = {}; pose = {}; });
    toast('已恢复 CAD 零位。');
  });
}

function renderAI(target) {
  const result = aiResult;
  const suggestions = result?.suggestions ?? [];
  target.innerHTML = heading('结构分析', 'Link 建议', 'ai') +
    '<p class="subtle">按装配路径与几何特征为未分配零件提出 Link 分组。确认后可应用到当前工程。</p>' +
    '<button class="button primary wide" id="analyze">' + icon('sparkles') + '分析结构</button>' +
    '<button class="button quiet wide" id="downloadStructure">' + icon('file-code') + '导出结构 JSON</button>' +
    (result ? '<div class="provider">来源：' + esc(result.provider) + '</div>' : '') +
    (result && !suggestions.length ? '<div class="empty-tree">没有待分组的零件。</div>' : '') +
    suggestions.map((item, index) => '<div class="suggestion"><div><strong>' + esc(item.name) + '</strong><span>' +
      item.parts.length + ' 件</span></div><p>' + esc(item.reason) + '</p><button class="text-button" data-apply="' + index +
      '">应用建议 ' + icon('arrow-right') + '</button></div>').join('');
  $('#analyze').onclick = async event => {
    event.currentTarget.disabled = true;
    event.currentTarget.textContent = '正在分析…';
    try {
      aiResult = await (await projectApi('suggestions')).json();
    } catch (error) { toast(error.message, true); }
    renderInspector();
  };
  $('#downloadStructure').onclick = async () => {
    try {
      const response = await projectApi('structure');
      download(await response.blob(), project.name + '-structure.json');
    } catch (error) { toast(error.message, true); }
  };
  target.querySelectorAll('[data-apply]').forEach(button => button.onclick = () => {
    const item = suggestions[Number(button.dataset.apply)];
    const assigned = new Set(project.links.flatMap(link => link.parts));
    const available = item.parts.filter(key => parts[key] && !assigned.has(key));
    if (!available.length) return toast('这些零件已分配。');
    createLink(item.name, available, item.origin);
    aiResult = { ...result, suggestions: suggestions.filter(candidate => candidate !== item) };
    inspector = 'ai';
    renderInspector();
  });
}

function modal(title, content, css = '') {
  dialog.className = css;
  dialog.innerHTML = '<div class="dialog-header"><h2 id="dialogTitle">' + esc(title) +
    '</h2><button id="closeDialog" class="icon-button" aria-label="关闭">' + icon('x') + '</button></div>' + content;
  $('#closeDialog').onclick = () => dialog.close();
  if (!dialog.open) dialog.showModal();
  paintIcons();
}

function formDialog(title, fields, confirm, onSubmit) {
  modal(title, '<form id="dialogForm" class="dialog-body form-body">' + fields +
    '<p id="formError" class="form-error" hidden></p><div class="dialog-actions">' +
    '<button id="cancelDialog" type="button" class="button secondary">取消</button><button class="button primary" type="submit">' +
    confirm + '</button></div></form>');
  $('#cancelDialog').onclick = () => dialog.close();
  $('#dialogForm').onsubmit = event => {
    event.preventDefault();
    const values = Object.fromEntries([...event.currentTarget.querySelectorAll('[data-field]')].map(input =>
      [input.dataset.field, input.type === 'number' ? input.valueAsNumber : input.value]));
    for (const input of event.currentTarget.querySelectorAll('[data-vector]')) {
      (values[input.dataset.vector] ??= [0, 0, 0])[Number(input.dataset.coordinate)] = input.valueAsNumber;
    }
    try {
      onSubmit(values);
      dialog.close();
    } catch (error) {
      $('#formError').textContent = error.message;
      $('#formError').hidden = false;
    }
  };
}

function nameOf(value) {
  const name = value.trim();
  if (!name) throw new Error('请输入名称。');
  return name;
}

function uniqueName(base, entries) {
  const names = new Set(entries.map(item => item.name));
  let name = base;
  let number = 2;
  while (names.has(name)) name = base + '_' + number++;
  return name;
}

function createLink(name, ids, origin) {
  const link = { id: id(), name: uniqueName(nameOf(name), project.links), parts: ids, origin,
    rpy: [...zero], density: 1000, mass: null, center: null, inertia: null, collision: 'mesh' };
  edit(() => {
    for (const existing of project.links) existing.parts = existing.parts.filter(key => !ids.includes(key));
    project.links.push(link);
    treeTab = 'links';
    selection = { kind: 'links', id: link.id };
    inspector = 'properties';
    chosenParts.clear();
  });
}

function openLink() {
  const ids = [...chosenParts];
  const available = ids.length ? ids : [];
  const origin = available.length ? zero.map((_, axis) =>
    available.reduce((value, key) => value + parts[key].center[axis], 0) / available.length) : [...zero];
  const name = available.length === 1 ? parts[available[0]].name : 'link_' + (project.links.length + 1);
  formDialog('创建 Link', field('Link 名称', 'name', name) + '<p class="subtle">已选 ' + available.length +
    ' 个零件。可在零件列表中继续分配。</p>' + vectors('原点 (m)', 'origin', origin), '创建',
  values => {
    if (!values.origin.every(Number.isFinite)) throw new Error('请输入有效坐标。');
    createLink(values.name, available, values.origin);
  });
}

function openJoint() {
  if (project.links.length < 2) return toast('请先创建至少两个 Link。', true);
  const parent = project.links[0];
  const child = project.links.find(link => link !== parent && !project.joints.some(joint => joint.child === link.id)) ?? project.links[1];
  const name = uniqueName(child.name + '_joint', project.joints);
  formDialog('创建关节', field('名称', 'name', name) + select('父 Link', 'parent', parent.id, linkOptions()) +
    select('子 Link', 'child', child.id, linkOptions()) + select('类型', 'type', 'revolute', Object.entries(kindLabel)) +
    '<p class="subtle">默认锚点取自所选子 Link，创建后可在检查器调整。</p>', '创建', values => {
      if (!allowableJoint(values.parent, values.child)) throw new Error('该连接会形成循环、重复父关节或自连接。');
      const target = project.links.find(link => link.id === values.child);
      const joint = { id: id(), name: uniqueName(nameOf(values.name), project.joints), parent: values.parent,
        child: target.id, type: values.type, anchor: [...target.origin], axis: [...unitZ], lower: -1.57,
        upper: 1.57, effort: 25, velocity: 3, damping: 0.1, friction: 0 };
      if (joint.type === 'prismatic') { joint.lower = -0.05; joint.upper = 0.05; }
      edit(() => {
        project.joints.push(joint);
        selection = { kind: 'joints', id: joint.id };
        treeTab = 'joints';
        inspector = 'properties';
      });
    });
}

function openLoop() {
  if (project.links.length < 2) return toast('请先创建至少两个 Link。', true);
  formDialog('创建闭链约束', field('名称', 'name', 'loop_' + (project.loops.length + 1)) +
    select('第一个 Link', 'first', project.links[0].id, linkOptions()) +
    select('第二个 Link', 'second', project.links[1].id, linkOptions()) +
    select('约束类型', 'type', 'connect', [['connect', '球形连接'], ['weld', '刚性焊接']]) +
    vectors('锚点 (m)', 'anchor', project.links[1].origin), '创建', values => {
      if (values.first === values.second || !values.anchor.every(Number.isFinite)) throw new Error('请选择两个不同的 Link 并填写锚点。');
      values.name = uniqueName(nameOf(values.name), project.loops);
      edit(() => {
        project.loops.push(values);
        selection = { kind: 'loops', id: values.name };
        inspector = 'properties';
      });
    });
}

function openMimic(joint) {
  const candidates = project.joints.filter(entry =>
    entry.id !== joint.id && ['revolute', 'continuous', 'prismatic'].includes(entry.type));
  formDialog('关节耦合', select('主动关节', 'joint', joint.mimic?.joint ?? '',
    [['', '独立驱动'], ...candidates.map(entry => [entry.id, entry.name])]) +
    field('倍率', 'multiplier', joint.mimic?.multiplier ?? 1, 'number') +
    field('偏移 (m 或 rad)', 'offset', joint.mimic?.offset ?? 0, 'number'), '应用', values => {
      if (!Number.isFinite(values.multiplier) || !Number.isFinite(values.offset)) throw new Error('请输入有效倍率和偏移。');
      const seen = new Set([joint.id]);
      let target = values.joint;
      while (target) {
        if (seen.has(target)) throw new Error('该耦合会形成循环。');
        seen.add(target);
        target = project.joints.find(entry => entry.id === target)?.mimic?.joint;
      }
      edit(() => { joint.mimic = values.joint ? values : null; });
    });
}

function openMass(link, physical) {
  const inertia = link.inertia ?? physical?.inertia ?? [];
  formDialog('质量与惯量', '<p class="subtle">留空时使用零件几何与密度计算。惯量需在 Link 坐标系下给出。</p>' +
    field('质量 (kg)', 'mass', link.mass ?? '', 'number') +
    ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz'].map((label, index) =>
      field(label + ' (kg·m²)', 'i' + index, link.inertia ? inertia[index] : '', 'number')).join('') +
    '<div id="customCenter"></div>' + vectors('质心 (m)', 'center', link.center ?? physical?.center ?? zero) +
    '<label class="check-row"><input type="checkbox" id="customCenterToggle"' + (link.center ? ' checked' : '') + '>手动指定质心</label>', '应用', values => {
      const entries = Array.from({ length: 6 }, (_, index) => values['i' + index]);
      const filled = entries.filter(Number.isFinite).length;
      if ((Number.isFinite(values.mass) && values.mass <= 0) || (filled !== 0 && filled !== 6) ||
        ($('#customCenterToggle').checked && !values.center.every(Number.isFinite))) throw new Error('请输入正质量、完整惯量和有效质心。');
      edit(() => {
        link.mass = Number.isFinite(values.mass) ? values.mass : null;
        link.inertia = filled ? entries : null;
        link.center = $('#customCenterToggle').checked ? values.center : null;
      });
    });
}

function openSettings() {
  formDialog('模型基准与初始位姿', field('机器人名称', 'name', project.name) +
    vectors('平移 (m)', 'world_origin', project.world_origin) + vectors('旋转 RPY (rad)', 'world_rpy', project.world_rpy) +
    select('仿真底座', 'fixed_base', project.fixed_base ? 'fixed' : 'floating', [['fixed', '固定'], ['floating', '自由']]),
  '应用', values => {
    if (![...values.world_origin, ...values.world_rpy].every(Number.isFinite)) throw new Error('请输入有效位姿。');
    const name = nameOf(values.name);
    edit(() => Object.assign(project, values, { name, fixed_base: values.fixed_base === 'fixed' }));
  });
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

async function saveProject() {
  if (!job) return;
  try {
    await cacheProject(false);
    const data = JSON.stringify({ ...job, project });
    download(new Blob([data], { type: 'application/json' }), project.name + '.cad2urdf.json');
    toast('工程已保存到缓存并下载。');
  } catch (error) { toast(error.message, true); }
}

async function openImport() {
  modal('导入模型', '<div class="dialog-body"><div class="drop-area" id="dropArea">' + icon('cloud-upload') +
    '<strong>拖放模型或资源 ZIP</strong><span>STEP / STP · URDF · MJCF · USD · 工程文件</span><div class="inline-actions">' +
    '<button id="pickFiles" class="button primary">选择文件</button><button id="pickFolder" class="button secondary">选择文件夹</button>' +
    '</div></div><p id="importMessage" class="form-error" hidden></p><div class="dialog-split"><section><h4>内置示例</h4>' +
    '<button class="button secondary wide" data-demo="arm">' + icon('box') + 'Atlas R6 机械臂</button>' +
    '<button class="button secondary wide" data-demo="rover">' + icon('box') + 'Scout 4WD 小车</button></section>' +
    '<section><h4>本地工程缓存</h4><div id="cachedProjects" class="cache-list">正在读取…</div></section></div></div>', 'import-dialog');
  $('#pickFiles').onclick = () => $('#modelInput').click();
  $('#pickFolder').onclick = () => $('#folderInput').click();
  const drop = $('#dropArea');
  drop.ondragover = event => { event.preventDefault(); drop.classList.add('drag'); };
  drop.ondragleave = () => drop.classList.remove('drag');
  drop.ondrop = async event => {
    event.preventDefault();
    drop.classList.remove('drag');
    const items = [...event.dataTransfer.items];
    const files = [...event.dataTransfer.files];
    try { await importFiles(await droppedFiles(items, files)); }
    catch (error) { toast(error.message, true); }
  };
  dialog.querySelectorAll('[data-demo]').forEach(button => button.onclick = () => loadDemo(button.dataset.demo));
  try {
    const cached = await (await api('/projects')).json();
    const list = $('#cachedProjects');
    if (!list) return;
    list.innerHTML = cached.length ? cached.map(item => '<button data-cache="' + esc(item.id) + '">' +
      icon('folder') + '<span>' + esc(item.name) + '</span></button>').join('') : '<span class="subtle">暂无已保存工程</span>';
    list.querySelectorAll('button').forEach(button => button.onclick = async () => {
      try { load(await (await api('/projects/' + button.dataset.cache)).json()); }
      catch (error) { toast(error.message, true); }
    });
    paintIcons();
  } catch (error) { toast(error.message, true); }
}

async function droppedFiles(items, fallback) {
  const entries = items.map(item => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.some(entry => entry.isDirectory)) return fallback;
  const result = [];
  async function walk(entry) {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      result.push({ file, path: entry.fullPath.replace(/^\//, '') });
      return;
    }
    const reader = entry.createReader();
    while (true) {
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) await walk(child);
    }
  }
  for (const entry of entries) await walk(entry);
  return result;
}

async function loadDemo(name) {
  if (loading) return;
  loading = true;
  try {
    load(await (await api('/demo/' + name, 'POST')).json());
    latestFiles = [];
    preferredModel = '';
  } catch (error) { toast(error.message, true); }
  finally { loading = false; }
}

async function importFiles(files) {
  if (!files.length || loading) return;
  if (!dialog.open) await openImport();
  loading = true;
  const status = $('#importMessage');
  const body = new FormData();
  const newFiles = files.map(entry => ({ file: entry.file ?? entry, path: entry.path ?? (entry.webkitRelativePath || entry.name) }));
  const main = newFiles.find(item => /\.(urdf|stp|step|usd[ac]?|mjcf|xml)$/i.test(item.path));
  const augment = latestFiles.length && newFiles.some(item => item.path.includes('/')) &&
    (!main || !newFiles.some(item => /\.(step|stp|urdf|usd[ac]?|mjcf)$/i.test(item.path)));
  const merged = new Map((augment ? latestFiles : []).map(item => [item.path.toLowerCase(), item]));
  newFiles.forEach(item => merged.set(item.path.toLowerCase(), item));
  latestFiles = [...merged.values()];
  if (!augment) preferredModel = newFiles.length === 1 ? (main?.path ?? '') : '';
  latestFiles.forEach(item => body.append('files', item.file, item.path));
  body.set('preferred', preferredModel);
  status.hidden = false;
  status.classList.remove('form-error');
  status.textContent = '正在解析几何与装配结构…';
  try {
    load(await (await api('/import', 'POST', body)).json());
    latestFiles = [];
    preferredModel = '';
    toast('模型已导入。');
  } catch (error) {
    status.classList.add('form-error');
    status.textContent = error.message;
  } finally { loading = false; }
}

function openExport() {
  if (!project) return;
  step = 'export';
  render();
  modal('测试与导出', '<div class="dialog-body export-body"><div class="export-summary">' +
    '<span class="selection-symbol">' + icon('box') + '</span><div><strong>' + esc(project.name) +
    '</strong><p>' + project.links.length + ' Link · ' + project.joints.length + ' 关节</p></div></div>' +
    '<section><h4>模型格式</h4><div class="format-options">' +
    [['urdf', 'URDF', 'ROS / 通用'], ['mjcf', 'MJCF', 'MuJoCo'], ['usd', 'USD', 'Isaac Sim']].map(([value, name, description]) =>
      '<label><input name="format" type="checkbox" value="' + value + '"' + (exportOptions.formats.includes(value) ? ' checked' : '') +
      '><strong>' + name + '</strong><span>' + description + '</span></label>').join('') + '</div>' +
    '<label class="check-row"><input id="mergeMeshes" type="checkbox"' + (exportOptions.merge ? ' checked' : '') +
    '>每个 Link 合并导出的 STL 网格</label>' +
    '<div class="inline-actions"><button class="button secondary" id="verifyModel">' + icon('check') + '检查模型</button>' +
    '<button class="button secondary" id="testPose">' + icon('scan-line') + '在 3D 中试转</button>' +
    '<button class="button primary push" id="downloadModel">' + icon('download') + '下载模型 ZIP</button></div></section>' +
    '<section class="training-panel"><div><div class="eyebrow">强化学习</div><h4>生成训练工程</h4>' +
    '<p class="subtle">生成关节姿态跟踪 PPO 示例，包含模型、训练脚本和运行说明。</p></div>' +
    select('训练框架', 'training', exportOptions.training, [['isaac', 'Isaac Lab'], ['mjlab', 'MJLab']]) +
    '<button class="button secondary wide" id="downloadTraining">' + icon('file-code') + '下载训练工程 ZIP</button></section>' +
    '<div id="exportStatus" class="export-status" aria-live="polite" hidden></div></div>', 'export-dialog');
  dialog.querySelectorAll('[name="format"]').forEach(input => input.onchange = () => {
    exportOptions.formats = [...dialog.querySelectorAll('[name="format"]:checked')].map(element => element.value);
  });
  $('#mergeMeshes').onchange = event => { exportOptions.merge = event.currentTarget.checked; };
  dialog.querySelector('[data-field="training"]').onchange = event => { exportOptions.training = event.currentTarget.value; };
  $('#testPose').onclick = () => { inspector = 'pose'; dialog.close(); render(); };
  $('#verifyModel').onclick = () => verifyModel();
  $('#downloadModel').onclick = () => downloadExport();
  $('#downloadTraining').onclick = () => downloadExport(exportOptions.training);
}

function showStatus(messages, error = false) {
  const target = $('#exportStatus');
  if (!target) return;
  target.hidden = false;
  target.classList.toggle('error', error);
  target.textContent = messages.join('\n');
}

async function verifyModel() {
  try {
    const data = await (await projectApi('validate', { project, formats: exportOptions.formats, merge: exportOptions.merge })).json();
    showStatus([...(data.errors.length ? data.errors : ['模型校验通过。']), ...data.warnings], Boolean(data.errors.length));
  } catch (error) { showStatus([error.message], true); }
}

async function downloadExport(training = null) {
  const buttons = [$('#downloadModel'), $('#downloadTraining')];
  buttons.forEach(button => { button.disabled = true; });
  showStatus(['正在生成导出包…']);
  try {
    const response = await projectApi('export', { project, formats: training ? [] : exportOptions.formats,
      merge: exportOptions.merge, training });
    const filename = response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] ?? 'robot.zip';
    download(await response.blob(), filename);
    showStatus([training ? '训练工程已生成。解压后查看 TRAINING.md。' : '模型已打包。']);
  } catch (error) { showStatus([error.message], true); }
  finally { buttons.forEach(button => { button.disabled = false; }); }
}

function resetPose() {
  pose = {};
  renderInspector();
  syncView();
}

for (const button of document.querySelectorAll('[data-step]')) button.onclick = () => {
  if (button.dataset.step === 'import') return openImport();
  if (button.dataset.step === 'export') return openExport();
  step = 'configure';
  render();
};
for (const button of document.querySelectorAll('[data-tree]')) button.onclick = () => {
  treeTab = button.dataset.tree;
  renderTree();
};
for (const button of document.querySelectorAll('[data-inspect]')) button.onclick = () => {
  inspector = button.dataset.inspect;
  renderInspector();
};
$('#search').oninput = () => { if (project) renderTree(); };
$('#importButton').onclick = openImport;
$('#openExport').onclick = openExport;
$('#saveProject').onclick = saveProject;
$('#undo').onclick = () => history(true);
$('#redo').onclick = () => history(false);
$('#fit').onclick = () => viewer?.fit();
$('#topView').onclick = () => viewer?.fit(true);
$('#resetPose').onclick = () => { if (project) resetPose(); };
$('#showCollisions').onclick = event => {
  if (!project) return;
  collisions = !collisions;
  event.currentTarget.setAttribute('aria-pressed', collisions);
  syncView();
};
$('#help').onclick = () => modal('工作台帮助', '<div class="dialog-body help-body"><ol>' +
  '<li>导入 STEP 或已有机器人模型。缺少资源时补选模型文件夹。</li>' +
  '<li>在零件列表或 3D 中选择零件，创建 Link。按 Shift 或 Ctrl 多选。</li>' +
  '<li>创建关节，检查父子关系、转轴和限位。若机构有闭环，添加单独的闭链约束。</li>' +
  '<li>在试转页检查运动，在测试与导出中生成模型或训练工程。</li></ol></div>');
for (const input of [$('#modelInput'), $('#folderInput')]) input.onchange = () => {
  const files = [...input.files];
  input.value = '';
  importFiles(files);
};
document.addEventListener('keydown', event => {
  if (!project || dialog.open || !(event.ctrlKey || event.metaKey) || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (key === 'z' || key === 'y') { event.preventDefault(); history(key === 'z' && !event.shiftKey); }
});
paintIcons();
const previous = localStorage.getItem('step2urdf.last');
try {
  if (!previous) await loadDemo('arm');
  else load(await (await api('/projects/' + previous)).json());
} catch {
  await loadDemo('arm');
}
if (!job) {
  $('#viewMessage').textContent = '无法连接转换服务。请启动本地服务后重新加载。';
}
