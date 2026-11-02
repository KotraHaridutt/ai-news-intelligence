// Tailwind v3 (the version the page's Play CDN served), built by scripts/build_css.py.
// Every file that names Tailwind classes: the page and its script (class names are written out in
// full there, e.g. "border-blue-600 text-blue-700", so the build finds them). Paths are from the repo root.
module.exports = {
    content: ["./templates/**/*.html", "./static/app.js"],
    theme: { extend: {} },
    plugins: [],
};
