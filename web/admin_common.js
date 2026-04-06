const ADMIN_TOKEN_KEY = "adminToken";
const ADMIN_USER_KEY = "adminUser";

/** 未登录可访问（仅此白名单） */
const AUTH_PUBLIC_PATHS = new Set(["/login.html", "/student_register.html"]);

function normalizePathname() {
  let p = window.location.pathname || "/";
  if (!p.startsWith("/")) p = "/" + p;
  const q = p.indexOf("?");
  if (q >= 0) p = p.slice(0, q);
  return p.toLowerCase();
}

function isAuthPublicPath(p) {
  return AUTH_PUBLIC_PATHS.has(p);
}

/** 管理后台 HTML：/admin_*.html */
function isAdminAppPath(p) {
  return /\/admin_[^/]+\.html$/i.test(p);
}

function getAdminToken() {
  return localStorage.getItem(ADMIN_TOKEN_KEY) || sessionStorage.getItem(ADMIN_TOKEN_KEY) || "";
}

function isAdminLoggedIn() {
  return !!getAdminToken();
}

function setAdminLogin(token, username, remember) {
  const t = token || "1";
  localStorage.removeItem(ADMIN_TOKEN_KEY);
  localStorage.removeItem(ADMIN_USER_KEY);
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
  sessionStorage.removeItem(ADMIN_USER_KEY);

  if (remember) {
    localStorage.setItem(ADMIN_TOKEN_KEY, t);
    if (username) localStorage.setItem(ADMIN_USER_KEY, username);
  } else {
    sessionStorage.setItem(ADMIN_TOKEN_KEY, t);
    if (username) sessionStorage.setItem(ADMIN_USER_KEY, username);
  }
}

function logoutAdmin() {
  localStorage.removeItem(ADMIN_TOKEN_KEY);
  localStorage.removeItem(ADMIN_USER_KEY);
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
  sessionStorage.removeItem(ADMIN_USER_KEY);
  window.location.href = "/login.html";
}

/** 登录后跳转：学生不可去 admin 路径 */
function authSafeNextUrl(role, nextRaw) {
  if (!nextRaw || typeof nextRaw !== "string") return null;
  let path = nextRaw.trim();
  if (!path.startsWith("/") || path.startsWith("//")) return null;
  const low = path.split("?")[0].toLowerCase();
  if (role !== "admin" && isAdminAppPath(low)) return null;
  if (isAuthPublicPath(low)) return null;
  return path;
}

function defaultHomeForRole(role) {
  return role === "admin" ? "/admin_digitals.html" : "/dashboard.html";
}

/** 同步：无 token 且非公开页 → 立即去登录 */
function authGateSync() {
  try {
    const p = normalizePathname();
    if (isAuthPublicPath(p)) return;
    if (!getAdminToken()) {
      const next = encodeURIComponent(p + (window.location.search || ""));
      window.location.replace("/login.html?next=" + next);
    }
  } catch (_) {}
}

/**
 * 异步：校验 token；学生访问管理员页面 → 主页；非法 token → 登出
 */
async function authGateAsync() {
  const p = normalizePathname();
  if (isAuthPublicPath(p)) return;
  if (!getAdminToken()) return;
  let me;
  try {
    me = await apiGet("/auth/me");
  } catch (_) {
    me = null;
  }
  if (!me || me.code !== 0 || !me.data) {
    logoutAdmin();
    return;
  }
  const role = me.data.role;
  if (isAdminAppPath(p) && role !== "admin") {
    window.location.replace("/dashboard.html");
    return;
  }
}

function requireAdminLoginOrRedirect() {
  if (!isAdminLoggedIn()) {
    window.location.href = "/login.html";
  }
}

/**
 * 页面声明期望角色（当前用于 admin 页 expectedRole === "admin"）。
 * 学生误进：跳转学生主页，不清空 token。
 */
async function requireRoleOrRedirect(expectedRole) {
  if (!isAdminLoggedIn()) {
    window.location.href = "/login.html";
    return;
  }
  let me;
  try {
    me = await apiGet("/auth/me");
  } catch (_) {
    me = null;
  }
  if (!me || me.code !== 0 || !me.data) {
    logoutAdmin();
    return;
  }
  if (me.data.role !== expectedRole) {
    if (expectedRole === "admin") {
      window.location.replace("/dashboard.html");
      return;
    }
    window.location.replace(defaultHomeForRole(me.data.role));
    return;
  }
}

async function apiPost(url, body) {
  const token = getAdminToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const resp = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body || {}),
  });
  return resp.json();
}

async function apiGet(url) {
  const token = getAdminToken();
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const resp = await fetch(url, { method: "GET", headers });
  return resp.json();
}

function bindMenu(activePage) {
  const map = {
    digitals: document.getElementById("menu-digitals"),
    kb: document.getElementById("menu-kb"),
    sessions: document.getElementById("menu-sessions"),
    analytics: document.getElementById("menu-analytics"),
    students: document.getElementById("menu-students"),
  };
  for (const key of Object.keys(map)) {
    if (!map[key]) continue;
    map[key].classList.toggle("active", key === activePage);
  }
}

authGateSync();

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", function () {
    authGateAsync();
  });
}

// Load frontend i18n (file-based) for all pages that include admin_common.js.
// Student/public pages that don't include admin_common.js should add /i18n3.js manually.
(function () {
  if (typeof document === "undefined") return;
  if (window.__i18n3_loaded__) return;
  window.__i18n3_loaded__ = true;
  const s = document.createElement("script");
  s.src = "/i18n3.js";
  s.async = true;
  document.head.appendChild(s);
})();
