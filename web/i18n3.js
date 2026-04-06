(() => {
  // 防止同一页面被加载 i18n3.js 两次（静态 <script> + admin_common 动态注入）。
  // 双重初始化会导致切语言时事件/状态异常，表现为“必须刷新才生效”。
  if (typeof window !== "undefined") {
    if (window.__i18n3_initialized__) return;
    window.__i18n3_initialized__ = true;
  }
  const LANG_STORAGE_KEY = "uiLang";
  const LANGS = [
    { id: "zh-CN", label: "中文(简体)" },
    { id: "en", label: "English" },
    { id: "ja", label: "日本語" },
    { id: "ko", label: "한국어" },
  ];

  const I18N_DATA_URL = "/i18n/frontend_i18n.json";
  const SWITCHER_ID = "i18n-switcher";

  let data = null;
  let zhToTranslations = null;
  let zhRegex = null;
  let currentLang = "zh-CN";

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function isCJK(text) {
    return /[\u4e00-\u9fff]/.test(text);
  }

  function getCurrentLang() {
    const saved = localStorage.getItem(LANG_STORAGE_KEY);
    if (saved && LANGS.some((x) => x.id === saved)) return saved;
    const htmlLang = (document.documentElement && document.documentElement.lang) ? document.documentElement.lang : "";
    if (htmlLang && LANGS.some((x) => x.id === htmlLang)) return htmlLang;
    return "zh-CN";
  }

  function installUiLangHeaderForFetch() {
    if (typeof window === "undefined" || typeof window.fetch !== "function") return;
    if (window.__i18n3_fetch_patched__) return;
    window.__i18n3_fetch_patched__ = true;

    const rawFetch = window.fetch.bind(window);
    window.fetch = function patchedFetch(input, init) {
      const cfg = init ? { ...init } : {};
      const headers = new Headers(cfg.headers || {});
      const lang = getCurrentLang();
      if (lang) headers.set("X-UI-Lang", lang);
      cfg.headers = headers;
      return rawFetch(input, cfg);
    };
  }

  function getSwitcherLabel(lang) {
    switch (lang) {
      case "zh-CN":
        return "语言";
      case "en":
        return "Language";
      case "ja":
        return "言語";
      case "ko":
        return "언어";
      default:
        return "Language";
    }
  }

  function createSwitcher() {
    if (document.getElementById(SWITCHER_ID)) return;

    const wrap = document.createElement("div");
    wrap.id = SWITCHER_ID;
    wrap.style.position = "fixed";
    wrap.style.top = "12px";
    wrap.style.right = "12px";
    wrap.style.zIndex = "99999";
    wrap.style.background = "rgba(0,0,0,0.35)";
    wrap.style.border = "1px solid rgba(255,255,255,0.15)";
    wrap.style.backdropFilter = "blur(6px)";
    wrap.style.padding = "8px 10px";
    wrap.style.borderRadius = "10px";
    wrap.style.color = "white";
    wrap.style.fontSize = "13px";

    const label = document.createElement("span");
    label.textContent = getSwitcherLabel(currentLang);
    label.style.marginRight = "6px";
    wrap.appendChild(label);

    const sel = document.createElement("select");
    sel.style.background = "rgba(255,255,255,0.08)";
    sel.style.border = "1px solid rgba(255,255,255,0.15)";
    sel.style.color = "white";
    sel.style.borderRadius = "8px";
    sel.style.padding = "6px 8px";

    for (const l of LANGS) {
      const opt = document.createElement("option");
      opt.value = l.id;
      opt.textContent = l.label;
      sel.appendChild(opt);
    }

    sel.value = currentLang;
    sel.addEventListener("change", async () => {
      const next = sel.value || "zh-CN";
      localStorage.setItem(LANG_STORAGE_KEY, next);
      if (document && document.documentElement) document.documentElement.lang = next;
      currentLang = next;
      label.textContent = getSwitcherLabel(next);
      // User requested: switch language -> auto refresh page.
      // This avoids any partial/dynamic rendering issues on some pages.
      if (typeof window !== "undefined") {
        if (!window.__i18n3_reloading__) {
          window.__i18n3_reloading__ = true;
          setTimeout(() => window.location.reload(), 50);
        }
      }
    });

    wrap.appendChild(sel);
    document.body.appendChild(wrap);
  }

  function translateText(text) {
    if (!text || typeof text !== "string") return text;
    if (currentLang === "zh-CN") return text;
    if (!zhRegex) return text;
    if (!isCJK(text)) return text;
    return text.replace(zhRegex, (m) => {
      const pack = zhToTranslations[m];
      if (!pack) return m;
      return pack[currentLang] || m;
    });
  }

  function backupNodeText(node) {
    if (!node) return;
    if (node.__i18nOrigText === undefined) node.__i18nOrigText = node.nodeValue || "";
  }

  function restoreAll(root) {
    if (!root) return;
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          const tag = parent.tagName ? parent.tagName.toUpperCase() : "";
          if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") return NodeFilter.FILTER_REJECT;
          if (tag === "TEXTAREA" || tag === "INPUT") return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        },
      },
      false
    );

    let node;
    while ((node = walker.nextNode())) {
      if (node.__i18nOrigText !== undefined) node.nodeValue = node.__i18nOrigText;
    }

    const attrEls = root.querySelectorAll ? root.querySelectorAll("[placeholder],[title],[aria-label]") : [];
    for (const el of attrEls) {
      if (el.dataset.i18nOrigPlaceholder && el.getAttribute("placeholder") !== el.dataset.i18nOrigPlaceholder) {
        el.setAttribute("placeholder", el.dataset.i18nOrigPlaceholder);
      }
      if (el.dataset.i18nOrigTitle && el.getAttribute("title") !== el.dataset.i18nOrigTitle) {
        el.setAttribute("title", el.dataset.i18nOrigTitle);
      }
      if (el.dataset.i18nOrigAriaLabel && el.getAttribute("aria-label") !== el.dataset.i18nOrigAriaLabel) {
        el.setAttribute("aria-label", el.dataset.i18nOrigAriaLabel);
      }
    }

    const optionEls = root.querySelectorAll ? root.querySelectorAll("option") : [];
    for (const opt of optionEls) {
      if (opt.dataset.i18nOrigText != null) opt.textContent = opt.dataset.i18nOrigText;
    }

    if (document && document.__i18nOrigTitle) document.title = document.__i18nOrigTitle;
  }

  function applyAll(root) {
    if (!root) root = document.body;
    if (!zhToTranslations) return;

    function isSkipNode(node) {
      try {
        if (!node || !node.parentElement) return false;
        const parent = node.parentElement;
        const el1 = parent.closest && parent.closest('[data-i18n-skip-kb="1"]');
        const el2 = parent.closest && parent.closest('[data-i18n-skip-db="1"]');
        return !!el1 || !!el2;
      } catch {
        return false;
      }
    }

    // title
    if (document && document.title) {
      if (!document.__i18nOrigTitle) document.__i18nOrigTitle = document.title;
      document.title = translateText(document.title);
    }

    // attributes
    const attrEls = root.querySelectorAll ? root.querySelectorAll("[placeholder],[title],[aria-label]") : [];
    for (const el of attrEls) {
      if (el.closest && (el.closest('[data-i18n-skip-kb="1"]') || el.closest('[data-i18n-skip-db="1"]'))) continue;
      const ph = el.getAttribute("placeholder");
      if (ph != null && el.dataset.i18nOrigPlaceholder === undefined) el.dataset.i18nOrigPlaceholder = ph;
      const phSrc = (el.dataset.i18nOrigPlaceholder !== undefined) ? el.dataset.i18nOrigPlaceholder : ph;
      if (phSrc != null) el.setAttribute("placeholder", translateText(phSrc));

      const ti = el.getAttribute("title");
      if (ti != null && el.dataset.i18nOrigTitle === undefined) el.dataset.i18nOrigTitle = ti;
      const tiSrc = (el.dataset.i18nOrigTitle !== undefined) ? el.dataset.i18nOrigTitle : ti;
      if (tiSrc != null) el.setAttribute("title", translateText(tiSrc));

      const al = el.getAttribute("aria-label");
      if (al != null && el.dataset.i18nOrigAriaLabel === undefined) el.dataset.i18nOrigAriaLabel = al;
      const alSrc = (el.dataset.i18nOrigAriaLabel !== undefined) ? el.dataset.i18nOrigAriaLabel : al;
      if (alSrc != null) el.setAttribute("aria-label", translateText(alSrc));
    }

    // options
    const optionEls = root.querySelectorAll ? root.querySelectorAll("option") : [];
    for (const opt of optionEls) {
      if (opt.closest && (opt.closest('[data-i18n-skip-kb="1"]') || opt.closest('[data-i18n-skip-db="1"]'))) continue;
      if (opt.dataset.i18nOrigText === undefined) opt.dataset.i18nOrigText = opt.textContent || "";
      opt.textContent = translateText(opt.dataset.i18nOrigText || "");
    }

    // text nodes
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          const tag = parent.tagName ? parent.tagName.toUpperCase() : "";
          if (tag === "SCRIPT" || tag === "STYLE" || tag === "NOSCRIPT") return NodeFilter.FILTER_REJECT;
          if (tag === "TEXTAREA" || tag === "INPUT") return NodeFilter.FILTER_REJECT;
          const v = node.nodeValue || "";
          if (!v.trim() || !isCJK(v)) return NodeFilter.FILTER_REJECT;
          if (!zhRegex) return NodeFilter.FILTER_REJECT;
          if (isSkipNode(node)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        },
      },
      false
    );

    let node;
    while ((node = walker.nextNode())) {
      backupNodeText(node);
      const oldVal = node.nodeValue || "";
      // Critical: always translate from backed-up original zh text,
      // otherwise switching between non-zh languages won't update.
      const src = node.__i18nOrigText !== undefined ? node.__i18nOrigText : oldVal;
      const nv = translateText(src);
      if (nv !== oldVal) node.nodeValue = nv;
    }
  }

  function setupObserver() {
    if (!window.MutationObserver) return;
    const obs = new MutationObserver((muts) => {
      if (currentLang === "zh-CN") return;
      for (const m of muts) {
        for (const n of m.addedNodes || []) {
          if (n && n.nodeType === 1) applyAll(n);
        }
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }

  async function loadData() {
    const resp = await fetch(I18N_DATA_URL, { cache: "no-store" });
    if (!resp.ok) {
      console.warn("[i18n3] failed to load:", I18N_DATA_URL, resp.status);
      return false;
    }
    const json = await resp.json();
    data = json;
    const strings = json && json.strings && typeof json.strings === "object" ? json.strings : {};

    // zhToTranslations: { "用户中心": {en,ja,ko} }
    zhToTranslations = {};
    const zhKeys = Object.keys(strings);
    for (const zh of zhKeys) {
      const rec = strings[zh] || {};
      zhToTranslations[zh] = {
        en: rec.en || zh,
        ja: rec.ja || zh,
        ko: rec.ko || zh,
      };
    }

    const keysSorted = zhKeys
      .filter((x) => x && x.length >= 2)
      .sort((a, b) => b.length - a.length);
    if (keysSorted.length) {
      const pattern = keysSorted.map(escapeRegExp).join("|");
      zhRegex = new RegExp(pattern, "g");
    }
    return true;
  }

  async function init() {
    currentLang = getCurrentLang();
    installUiLangHeaderForFetch();
    createSwitcher();
    const ok = await loadData();
    if (!ok) return;
    if (currentLang !== "zh-CN") applyAll(document.body);
    setupObserver();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => init());
  } else {
    init();
  }
})();

