/**
 * Agentic Shopping Assistant — "Complete the Look" widget loader
 * Version: 1.0.0
 *
 * USAGE — paste exactly this one line anywhere on your product page:
 *
 *   <script src="https://asa-stylist.vercel.app/widget.js" data-brand="snitch" async></script>
 *
 * Optional attributes:
 *   data-brand="snitch"          — brand id (snitch | myntra | flipkart)
 *   data-origin="https://asa-stylist.vercel.app"  — override embed origin (default: script host)
 *   data-label="Complete the Look"               — button label
 *   data-accent="#1a1a2e"        — button accent colour (hex)
 *
 * The script is idempotent: safe to include twice on the same page.
 * No dependencies. No build step. Vanilla JS (ES2017+).
 */

(function () {
  "use strict";

  // Idempotency guard — skip if already initialised.
  if (window.__asaWidgetLoaded) return;
  window.__asaWidgetLoaded = true;

  // ---------------------------------------------------------------------------
  // Read config from <script> attributes
  // ---------------------------------------------------------------------------
  var scriptEl =
    document.currentScript ||
    (function () {
      var scripts = document.getElementsByTagName("script");
      return scripts[scripts.length - 1];
    })();

  var brand = scriptEl.getAttribute("data-brand") || "snitch";
  var scriptSrc = scriptEl.src || "";
  var scriptOrigin = scriptSrc
    ? scriptSrc.split("/").slice(0, 3).join("/")
    : window.location.origin;
  var embedOrigin = scriptEl.getAttribute("data-origin") || scriptOrigin;
  var buttonLabel = scriptEl.getAttribute("data-label") || "Complete the Look";
  var accentColour = scriptEl.getAttribute("data-accent") || "#1a1a2e";

  var EMBED_URL = embedOrigin + "/embed/" + encodeURIComponent(brand);

  // ---------------------------------------------------------------------------
  // Inject minimal global styles (scoped to .asa-widget-* classes)
  // ---------------------------------------------------------------------------
  var styleId = "asa-widget-styles";
  if (!document.getElementById(styleId)) {
    var style = document.createElement("style");
    style.id = styleId;
    style.textContent = [
      ".asa-trigger-btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;",
      "border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;",
      "transition:opacity 0.15s ease;font-family:inherit;}",
      ".asa-trigger-btn:hover{opacity:0.85;}",
      ".asa-trigger-btn:active{opacity:0.7;}",

      ".asa-overlay{position:fixed;inset:0;z-index:2147483647;",
      "background:rgba(0,0,0,0.55);display:flex;align-items:flex-end;",
      "justify-content:center;animation:asaFadeIn 0.2s ease;}",
      "@media(min-width:640px){.asa-overlay{align-items:center;}}",

      ".asa-modal{position:relative;background:#fff;border-radius:16px 16px 0 0;",
      "width:100%;max-width:480px;height:90vh;display:flex;flex-direction:column;",
      "overflow:hidden;animation:asaSlideUp 0.25s ease;}",
      "@media(min-width:640px){.asa-modal{border-radius:16px;height:85vh;}}",

      ".asa-modal-header{display:flex;align-items:center;justify-content:space-between;",
      "padding:12px 16px;border-bottom:1px solid #e5e7eb;background:#fff;shrink:0;}",
      ".asa-modal-title{font-size:13px;font-weight:600;color:#374151;font-family:inherit;}",
      ".asa-close-btn{display:flex;align-items:center;justify-content:center;",
      "width:28px;height:28px;border:none;border-radius:50%;background:#f3f4f6;",
      "cursor:pointer;font-size:14px;color:#6b7280;transition:background 0.15s;}",
      ".asa-close-btn:hover{background:#e5e7eb;}",

      ".asa-iframe{flex:1;width:100%;border:none;background:#f9fafb;}",

      "@keyframes asaFadeIn{from{opacity:0;}to{opacity:1;}}",
      "@keyframes asaSlideUp{from{transform:translateY(40px);opacity:0;}to{transform:translateY(0);opacity:1;}}",
    ].join("");
    document.head.appendChild(style);
  }

  // ---------------------------------------------------------------------------
  // Create trigger button and inject it
  // ---------------------------------------------------------------------------
  var triggerContainer = document.createElement("div");
  triggerContainer.id = "asa-widget-trigger";
  triggerContainer.style.cssText = "margin:12px 0;display:inline-block;";

  var triggerBtn = document.createElement("button");
  triggerBtn.className = "asa-trigger-btn";
  triggerBtn.setAttribute("aria-label", buttonLabel);
  triggerBtn.style.backgroundColor = accentColour;
  triggerBtn.style.color = "#ffffff";
  // Sparkle icon + label
  triggerBtn.innerHTML =
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" ' +
    'fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 ' +
    '7 14.14 2 9.27 8.91 8.26 12 2"/></svg>' +
    '<span>' + buttonLabel + '</span>';

  triggerContainer.appendChild(triggerBtn);

  // Try to inject after the page's add-to-cart button; fall back to body.
  function injectTrigger() {
    if (document.getElementById("asa-widget-trigger")) return; // idempotent
    var atcButton =
      document.querySelector('[class*="add-to-cart"]') ||
      document.querySelector('[id*="add-to-cart"]') ||
      document.querySelector('[class*="AddToCart"]') ||
      document.querySelector('[data-testid*="add-to-cart"]') ||
      document.querySelector('button[type="submit"]') ||
      document.body;
    if (atcButton && atcButton !== document.body && atcButton.parentNode) {
      atcButton.parentNode.insertBefore(triggerContainer, atcButton.nextSibling);
    } else {
      document.body.appendChild(triggerContainer);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", injectTrigger);
  } else {
    injectTrigger();
  }

  // ---------------------------------------------------------------------------
  // Modal / overlay logic
  // ---------------------------------------------------------------------------
  var overlay = null;

  function openModal() {
    if (overlay) return; // already open

    overlay = document.createElement("div");
    overlay.className = "asa-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", buttonLabel);

    var modal = document.createElement("div");
    modal.className = "asa-modal";

    // Header with title + close button
    var header = document.createElement("div");
    header.className = "asa-modal-header";

    var title = document.createElement("span");
    title.className = "asa-modal-title";
    title.textContent = buttonLabel;

    var closeBtn = document.createElement("button");
    closeBtn.className = "asa-close-btn";
    closeBtn.setAttribute("aria-label", "Close");
    closeBtn.textContent = "✕";
    closeBtn.onclick = closeModal;

    header.appendChild(title);
    header.appendChild(closeBtn);

    // Iframe
    var iframe = document.createElement("iframe");
    iframe.className = "asa-iframe";
    iframe.src = EMBED_URL;
    iframe.setAttribute(
      "sandbox",
      "allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox"
    );
    iframe.setAttribute("loading", "lazy");
    iframe.setAttribute("title", buttonLabel);
    iframe.setAttribute("referrerpolicy", "strict-origin-when-cross-origin");

    modal.appendChild(header);
    modal.appendChild(iframe);
    overlay.appendChild(modal);

    // Click outside the modal to close.
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeModal();
    });

    document.body.appendChild(overlay);
    document.body.style.overflow = "hidden";

    // Focus the iframe for keyboard accessibility.
    setTimeout(function () {
      iframe.focus();
    }, 250);
  }

  function closeModal() {
    if (!overlay) return;
    document.body.removeChild(overlay);
    document.body.style.overflow = "";
    overlay = null;
    triggerBtn.focus();
  }

  triggerBtn.addEventListener("click", openModal);

  // ---------------------------------------------------------------------------
  // Listen for postMessage close signal from the embed iframe
  // ---------------------------------------------------------------------------
  window.addEventListener("message", function (event) {
    // Only accept messages from our embed origin.
    if (event.origin !== embedOrigin) return;
    if (
      event.data &&
      typeof event.data === "object" &&
      event.data.type === "asa:close"
    ) {
      closeModal();
    }
  });
})();
