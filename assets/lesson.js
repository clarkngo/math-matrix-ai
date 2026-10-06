// Shared behavior for MathMatrix AI lesson pages: theme toggle, quiz feedback, section-nav highlight.
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  var btn = document.getElementById('themeBtn');
  if (btn) btn.textContent = (theme === 'dark') ? '☀️ Light' : '🌙 Dark';
  try { localStorage.setItem('mmx-theme', theme); } catch (e) {}
}
function toggleTheme() {
  applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}
(function () {
  var saved = null;
  try { saved = localStorage.getItem('mmx-theme'); } catch (e) {}
  var preset = document.documentElement.getAttribute('data-theme');
  var prefersDark = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || preset || (prefersDark ? 'dark' : 'light'));
})();

document.addEventListener('DOMContentLoaded', function () {
  applyTheme(document.documentElement.getAttribute('data-theme') || 'light');  // sync the button label once it exists

  // Quiz: each .q-card holds .q-opt buttons; the right one has data-correct, and data-explain on the card is shown after any pick.
  document.querySelectorAll('.q-card').forEach(function (card) {
    var fb = card.querySelector('.q-feedback');
    card.querySelectorAll('.q-opt').forEach(function (opt) {
      opt.addEventListener('click', function () {
        var right = opt.hasAttribute('data-correct');
        card.querySelectorAll('.q-opt').forEach(function (o) { o.classList.remove('right', 'wrong'); });
        opt.classList.add(right ? 'right' : 'wrong');
        if (right) {
          fb.innerHTML = '<strong>Correct.</strong> ' + card.getAttribute('data-explain');
        } else {
          fb.innerHTML = '<strong>Not quite.</strong> ' + (opt.getAttribute('data-why') || 'Try another option.');
        }
        if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([fb]);
      });
    });
  });

  // Highlight the section in view.
  var links = document.querySelectorAll('.section-nav a');
  if (!('IntersectionObserver' in window) || !links.length) return;
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      links.forEach(function (a) { a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id); });
    });
  }, { rootMargin: '-15% 0px -75% 0px' });
  document.querySelectorAll('main section[id]').forEach(function (s) { io.observe(s); });
});
