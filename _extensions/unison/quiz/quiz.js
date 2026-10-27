// Cuestionarios de autoevaluación del curso (sin servidor, sin almacenamiento).
(function () {
  "use strict";

  function aNumero(s) {
    if (s === null || s === undefined) return NaN;
    var t = String(s).trim().replace(/\s/g, "").replace(",", ".");
    // Admite fracciones simples "1/3"
    if (/^-?\d+(\.\d+)?\/\d+(\.\d+)?$/.test(t)) {
      var p = t.split("/");
      return parseFloat(p[0]) / parseFloat(p[1]);
    }
    return t === "" ? NaN : Number(t);
  }

  function evaluar(fs) {
    var tipo = fs.dataset.tipo;
    if (tipo === "numerica") {
      var v = aNumero(fs.querySelector(".rdp-quiz-numero").value);
      if (isNaN(v)) return null;
      var r = aNumero(fs.dataset.respuesta);
      var tol = aNumero(fs.dataset.tolerancia) || 0;
      return Math.abs(v - r) <= tol + 1e-12;
    }
    var entradas = fs.querySelectorAll("input");
    var alguna = false, ok = true;
    entradas.forEach(function (inp) {
      var correcta = inp.dataset.correcta === "1";
      if (inp.checked) alguna = true;
      if (inp.checked !== correcta) ok = false;
      var retro = inp.closest(".rdp-quiz-opcion").querySelector(".rdp-quiz-retro");
      if (retro) retro.hidden = !(inp.checked && retro.textContent.trim() !== "");
    });
    return alguna ? ok : null;
  }

  function actualizarMarcador(quiz) {
    var total = quiz.querySelectorAll(".rdp-quiz-pregunta").length;
    var bien = quiz.querySelectorAll(".rdp-quiz-pregunta.es-correcta").length;
    quiz.querySelector(".rdp-quiz-marcador").textContent = bien + " de " + total + " correctas";
  }

  function verificar(fs) {
    var res = evaluar(fs);
    var salida = fs.querySelector(".rdp-quiz-resultado");
    fs.classList.remove("es-correcta", "es-incorrecta");
    if (res === null) {
      salida.textContent = fs.dataset.tipo === "numerica" ? "Escribe un número." : "Elige una opción.";
      return;
    }
    fs.classList.add(res ? "es-correcta" : "es-incorrecta");
    salida.textContent = res ? "Correcto" : "Todavía no. Revisa y vuelve a intentar.";
    var expl = fs.querySelector(".rdp-quiz-explicacion");
    if (expl && res && expl.textContent.trim() !== "") expl.hidden = false;
    actualizarMarcador(fs.closest(".rdp-quiz"));
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([fs]).catch(function () {});
    }
  }

  function reiniciar(quiz) {
    quiz.querySelectorAll(".rdp-quiz-pregunta").forEach(function (fs) {
      fs.classList.remove("es-correcta", "es-incorrecta");
      fs.querySelectorAll("input").forEach(function (i) {
        if (i.type === "text") i.value = ""; else i.checked = false;
      });
      fs.querySelectorAll(".rdp-quiz-retro, .rdp-quiz-explicacion").forEach(function (e) { e.hidden = true; });
      fs.querySelector(".rdp-quiz-resultado").textContent = "";
    });
    actualizarMarcador(quiz);
  }

  document.addEventListener("click", function (ev) {
    var b = ev.target.closest(".rdp-quiz-verificar");
    if (b) { verificar(b.closest(".rdp-quiz-pregunta")); return; }
    var r = ev.target.closest(".rdp-quiz-reiniciar");
    if (r) reiniciar(r.closest(".rdp-quiz"));
  });

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && ev.target.classList && ev.target.classList.contains("rdp-quiz-numero")) {
      ev.preventDefault();
      verificar(ev.target.closest(".rdp-quiz-pregunta"));
    }
  });
})();
