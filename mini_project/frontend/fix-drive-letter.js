// Preload fix for Windows double drive letter in Metro bundler paths
// Metro on Windows creates paths like "D:\d:\project\..." when project is on lowercase "d:" drive
// This patches fs, path, and process.cwd at the lowest level to prevent the issue.
const fs = require('fs');
const _path = require('path');

// Fix any path with double drive letter: "X:\x:\..." -> "x:\..."
function fixDD(p) {
  if (typeof p === 'string' && p.length > 3) {
    // Pattern: single uppercase letter + :\ + single letter + :\
    if (p[1] === ':' && p[2] === '\\' && p[4] === ':' && p[5] === '\\') {
      return p.substring(2);
    }
    // Also handle forward slashes
    if (p[1] === ':' && p[2] === '/' && p[4] === ':' && p[5] === '/') {
      return p.substring(2);
    }
  }
  return p;
}

// Patch all fs functions
const fsFns = [
  'readFileSync','readFile','statSync','stat','existsSync',
  'realpathSync','realpath','accessSync','access',
  'lstatSync','lstat','openSync','open','readlinkSync','readlink',
  'readdirSync','readdir','mkdirSync','mkdir',
  'writeFileSync','writeFile','unlinkSync','unlink',
  'createReadStream','createWriteStream','chmodSync','chmod',
  'copyFileSync','copyFile','renameSync','rename',
  'rmdirSync','rmdir','truncateSync','truncate',
  'watchFile','unwatchFile','watch'
];

fsFns.forEach(fn => {
  if (fs[fn]) {
    const orig = fs[fn];
    fs[fn] = function(p, ...args) {
      return orig.call(this, fixDD(p), ...args);
    };
  }
});

// Also patch realpathSync.native if it exists
if (fs.realpathSync && fs.realpathSync.native) {
  const origNative = fs.realpathSync.native;
  fs.realpathSync.native = function(p, ...args) {
    return origNative.call(this, fixDD(p), ...args);
  };
}

// Patch path.resolve to never produce double drive paths
const origResolve = _path.resolve;
_path.resolve = function(...args) {
  const result = origResolve.apply(this, args);
  return fixDD(result);
};

const origJoin = _path.join;
_path.join = function(...args) {
  const result = origJoin.apply(this, args);
  return fixDD(result);
};

const origNormalize = _path.normalize;
_path.normalize = function(p) {
  return fixDD(origNormalize.call(this, p));
};
