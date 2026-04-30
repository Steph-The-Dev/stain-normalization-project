# 📘 Educational Report: Codebase Modernization & Engineering Best Practices
**Project:** Histological Stain Normalization Suite
**Target Audience:** MSc Data Science Students

This report explains the architectural and stylistic changes made to the repository. As a Data Science student, understanding these "Software Engineering for Data Science" (SE4DS) principles is crucial for building reproducible, maintainable, and production-ready pipelines.

---

## 1. Modular Architecture (The "Clean Code" Shift)
### Change:
Moved `batch_process.py` and `video_process.py` into `src/` and extracted UI utilities from `app.py` into `src/ui_utils.py`.

### Why?
*   **Separation of Concerns:** A monolithic file (like the old `app.py`) is hard to test and maintain. By separating the **UI (Streamlit)**, **Business Logic (Reinhard)**, and **Utilities (Parade rendering)**, we make the code modular.
*   **Reusability:** You can now import the normalization logic into a Jupyter Notebook or a separate CLI script without triggering the Streamlit UI.
*   **Clean Root Directory:** The root directory should only contain configuration files (`pyproject.toml`, `.gitignore`) and entry points. This makes the project structure professional at first glance.

---

## 2. Type Hinting & Type Safety
### Change:
Added type hints like `image: npt.NDArray[np.uint8]` and `return: Tuple[...]`.

### Why?
*   **Self-Documentation:** In Data Science, we often pass around NumPy arrays. Without type hints, you don't know if an array contains `float32` (normalized) or `uint8` (raw pixels). Hints make this explicit.
*   **Static Analysis:** Tools like MyPy can catch bugs before you even run the code (e.g., trying to pass a string where an image is expected).
*   **IDE Support:** VS Code and PyCharm use these hints to provide better autocomplete, which speeds up development.

---

## 3. Professional Documentation (NumPy Style)
### Change:
Translated comments to English and used the "NumPy Docstring" format.

### Why?
*   **Industry Standard:** The NumPy/Google docstring style is the standard in the scientific Python ecosystem (SciPy, Scikit-Learn, Pandas).
*   **Automated Docs:** This format allows tools like Sphinx to automatically generate professional HTML documentation from your code.
*   **Portfolio Presentation:** International recruiters and academic reviewers expect English. It demonstrates that your work is ready for a global professional context.

---

## 4. Modern Packaging & Dependency Management
### Change:
Introduced `pyproject.toml` and updated `environment.yml`.

### Why?
*   **Reproducibility:** This is the "Holy Grail" of Data Science. `pyproject.toml` (PEP 517/518) is the modern way to define exactly which versions of libraries your project needs.
*   **Mamba vs. Conda:** Conda's dependency solver can be extremely slow. **Mamba** uses a much faster solver (libsolv). For a master's student, saving 10 minutes every time you build an environment is a massive productivity gain.

---

## 5. From `print()` to `logging`
### Change:
Replaced `print()` with `import logging`.

### Why?
*   **Granularity:** Logging allows you to separate "Information" (e.g., "Processing image 5") from "Errors" (e.g., "Image corrupted").
*   **Production Readiness:** In a real production environment, you don't see `print()` outputs. Logs are instead redirected to files or monitoring services (like ELK or CloudWatch).
*   **Cleanliness:** You can turn off all "Info" logs with one line of code if you only want to see errors, which is impossible with `print()`.

---

## 6. The "DRY" Principle (Don't Repeat Yourself)
### Change:
Refactored `src/reinhard.py` to use an internal helper function `_apply_reinhard_stats`.

### Why?
*   **Maintainability:** Previously, the math for applying the normalization was copied in two different functions. If you found a bug in the math, you had to fix it in two places. Now, there is only one "source of truth."
*   **Readability:** It makes the public API (`normalize_stain_reinhard_hsv` and `normalize_stain_reinhard_luma`) much shorter and easier to understand.

---

### Summary for your MSc Portfolio:
By applying these changes, your project moved from a "collection of scripts" to a "Python Package." This demonstrates to professors and future employers that you not only understand the **Data Science (the math/CV)** but also the **Engineering (how to build reliable software)**.
