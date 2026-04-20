const path = require('path');
const fs = require('fs');

// ─── WINDOWS DRIVE LETTER FIX ────────────────────────────────────────────────
// Metro on Windows sometimes produces paths like "D:\d:\mini_project\..."
// (double drive letter). This monkey-patches fs methods to fix those paths
// before they reach the OS.
function fixDoubleDrive(p) {
  if (typeof p === 'string') {
    // Match patterns like "D:\d:\" or "C:\c:\" — a drive letter prefix duplicated
    const match = p.match(/^([A-Za-z]):\\([A-Za-z]):\\/);
    if (match) {
      return p.substring(2); // Remove the first "X:" prefix
    }
  }
  return p;
}

const originalReadFileSync = fs.readFileSync;
fs.readFileSync = function (filePath, ...args) {
  return originalReadFileSync.call(this, fixDoubleDrive(filePath), ...args);
};

const originalReadFile = fs.readFile;
fs.readFile = function (filePath, ...args) {
  return originalReadFile.call(this, fixDoubleDrive(filePath), ...args);
};

const originalStatSync = fs.statSync;
fs.statSync = function (filePath, ...args) {
  return originalStatSync.call(this, fixDoubleDrive(filePath), ...args);
};

const originalExistsSync = fs.existsSync;
fs.existsSync = function (filePath) {
  return originalExistsSync.call(this, fixDoubleDrive(filePath));
};

const originalRealpathSync = fs.realpathSync;
fs.realpathSync = function (filePath, ...args) {
  return originalRealpathSync.call(this, fixDoubleDrive(filePath), ...args);
};

const originalAccessSync = fs.accessSync;
fs.accessSync = function (filePath, ...args) {
  return originalAccessSync.call(this, fixDoubleDrive(filePath), ...args);
};

const originalLstatSync = fs.lstatSync;
fs.lstatSync = function (filePath, ...args) {
  return originalLstatSync.call(this, fixDoubleDrive(filePath), ...args);
};

// ─── METRO CONFIG ────────────────────────────────────────────────────────────
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

module.exports = config;
