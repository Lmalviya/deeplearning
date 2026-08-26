# Chapter 10 · Lesson 6 — Interview Lab: "Which Hyperparameter Tuning Method Have You Used?"

> **Where this fits:** This is the direct interview payoff of the whole chapter, and it addresses the exact worry you raised at the start of this chapter's discussion — that naming Bayesian optimization or Hyperband might sound more impressive than the honest answer. This lesson builds the case that the honest, small-grid-plus-inherited-priors answer is actually the *stronger* one, when explained with this chapter's reasoning behind it.

---

## 1. The Trap This Question Sets, and Why Overclaiming Fails It

An interviewer asking this question is very often testing for exactly the overclaiming risk Chapter 4, Lesson 7 warned about: a candidate who confidently says "I used Bayesian optimization with a Gaussian process surrogate" invites an immediate, specific follow-up — "what acquisition function, what kernel, how many trials, what was the search space" — that a fabricated or memorized-but-unused answer won't survive. Given this chapter's research, claiming Bayesian optimization or PBT for a large-model project is *also* likely to sound wrong to an interviewer who's actually run this kind of project themselves, since — per Lesson 4 — that's simply not how real large-scale tuning is usually done.

---

## 2. The Honest Answer, Built to Sound Credible Rather Than Underqualified

The key move: frame the small-grid-plus-inherited-priors approach as a **deliberate strategy**, not an absence of sophistication — because per this entire chapter, it genuinely is the sophisticated real-world approach.

> "For the scale I was working at, I didn't reach for something like Bayesian optimization or Hyperband — and that was a deliberate choice, not a limitation. The actual leverage in hyperparameter tuning at this scale comes from shrinking the search space before you search it at all: I anchored most hyperparameters — optimizer betas, weight decay, schedule shape — to values from closely related published work, since those are well-established enough that re-searching them wastes compute rediscovering something already known. For learning rate specifically, since it's the hyperparameter most sensitive to scale and task, I ran a small discrete grid — four or five log-spaced values — on a cheaper proxy setup, and picked the winner by validation performance before committing to the full-scale run. Given the search space was down to essentially one real dimension by that point, a small grid was just as effective as a more sophisticated search algorithm would have been, and far simpler to implement, debug, and fully parallelize."

---

## 3. Why This Answer Is Actually Stronger, Explicitly

Worth stating the comparison directly, since it's the whole point of this lesson: the naive assumption is "fancier algorithm = more impressive answer." The corrected understanding, built across this chapter: **knowing when a fancy algorithm is unnecessary, and why, is a stronger signal of practical experience than knowing the algorithm's mechanics in isolation.** A candidate who can explain *why* a small grid was sufficient — because the space had already been shrunk via known priors and structural techniques (Lesson 2) — demonstrates understanding of the actual economics of large-model training, which is precisely what separates someone who's read about HPO from someone who's run a real, expensive project under real constraints.

---

## 4. A Second Version, for When You DO Have Bayesian/Hyperband/PBT Experience

Worth including, since not every real situation matches the large-model case — if you've genuinely used one of Lesson 1's more sophisticated methods (e.g., Hyperband for a smaller model or a classical ML project, Lesson 5's territory), the honest answer there is different and should be stated with its own specific reasoning:

> "For a smaller-scale project — [specific example] — where individual trials were cheap enough to run dozens or hundreds of them, I used ASHA-style successive halving: starting a larger number of candidate configurations, training them for a small fraction of the full budget, and killing the bottom fraction before giving survivors more budget. That was the right call there specifically because early loss trajectory was a reasonably reliable predictor of final performance at that scale, and the cost savings from early termination were substantial given how many candidates I wanted to explore."

**Why having both versions ready matters:** the question "which method have you used" doesn't have one universally correct answer — it has a correct answer *for the specific project scale and constraints being discussed*, and demonstrating you'd choose differently for a different scale (per Lesson 1's comparison table) is itself a strong signal, arguably stronger than either answer given in isolation.

---

## 5. Follow-Up Questions to Have Pre-Loaded

**"Why didn't you just run Bayesian optimization anyway — wouldn't it have found a better result?"** → Direct callback to Lesson 1, Section 2: Bayesian optimization's sample-efficiency advantage shrinks as the search space shrinks, and its sequential nature is a real cost against a small, fully-parallelizable grid — for a space this small, the expected improvement wouldn't have justified the added implementation complexity and reduced parallelizability.

**"How did you decide which hyperparameters to inherit versus search?"** → Direct callback to Lesson 3: hyperparameters with strong, closely-related published priors (same rough model family, task, and scale) get inherited; only genuinely new or uncertain dimensions get searched — and this judgment call is itself explainable, not arbitrary.

**"What if the small-scale proxy's optimal LR didn't transfer well to the full-scale run — how would you know, and what would you do?"** → A genuine, honest limitation worth acknowledging directly: this is exactly the risk μP-style formal transfer (Chapter 4, Lesson 4; this chapter's Lesson 2) is designed to reduce versus a naive "just use the small-scale winner" approach — if formal transfer methodology wasn't used, a reasonable mitigation is validating with a short partial run at full scale before committing the entire budget, checking early loss behavior against Chapter 3, Lesson 8's expected-shape baseline before assuming the transferred LR is correct.

**"Have you ever used PBT, and if not, why not?"** → An honest answer per Lesson 1's usage table: PBT's requirement for a full population training simultaneously is resource-heavy in a way that's rarely justified outside of RL training or very large-scale vision projects specifically — for most LLM fine-tuning/alignment work, the fixed-schedule assumption (Chapter 3, Lesson 5's decay schedules) already captures most of the benefit PBT would provide for LR specifically, without needing a whole population's worth of compute.

---

## Key Takeaways

- This question tests whether small-grid-plus-inherited-priors is understood as a deliberate, sophisticated strategy or mistaken for a limitation — framing matters as much as the underlying facts.
- A strong answer explains *why* the search space was already small by the time a grid was applied, not just that a grid was used.
- Having a second, genuinely different answer ready for a different scale/constraint scenario (e.g., ASHA for a cheaper project) demonstrates the reasoning transfers, which is a stronger signal than a single memorized answer.
- Honest acknowledgment of transfer risk (proxy-to-full-scale) and of methods not used (PBT) — with a reasoned "why not" — is more credible than claiming universal expertise across every method in Lesson 1.

---

## Self-Check — Full Mock Rep

Construct your own honest answer using Section 2's structure, based on a real or realistic project of your own. Then have someone (or a future session with me) fire the four follow-ups from Section 5 at you, and practice the "different scale, different method" pivot from Section 4 if asked to contrast approaches.