const RANKS = [14,13,12,11,10,9,8,7,6,5,4,3,2];
const RANK_NAME = {14:'A',13:'K',12:'Q',11:'J',10:'T',9:'9',8:'8',7:'7',6:'6',5:'5',4:'4',3:'3',2:'2'};
const RED_SUITS  = ['♥','♦'];
const SUITS = ['♠','♥','♦','♣'];

const SCENARIO_SITUATIONS = {
  SB_RFI:     'SB — first to act preflop',
  BB_vs_RFI:  'BB — facing SB open',
  SB_vs_3bet: 'SB — facing BB 3-bet',
};
const CALL_LABELS  = { SB_RFI: 'LIMP', BB_vs_RFI: 'CALL', SB_vs_3bet: 'CALL' };
const RAISE_LABELS = { SB_RFI: 'RAISE', BB_vs_RFI: '3-BET', SB_vs_3bet: '4-BET' };
const OPP_LABELS   = { SB_RFI: 'BB', BB_vs_RFI: 'SB', SB_vs_3bet: 'BB' };
const MY_LABELS    = { SB_RFI: 'SB', BB_vs_RFI: 'BB', SB_vs_3bet: 'SB' };

let state = {
  rank1: 14, rank2: 13,
  suit1: '♠', suit2: '♥',
  suited: false,
  scenario: 'SB_RFI',
  mode: 'study',
  pickingSlot: null,
  pickingStage: null,
  quizAnswered: false,
  quizAnswer: null,
};

// ---- DOM refs ----
const rank1Display  = document.getElementById('rank1-display');
const rank2Display  = document.getElementById('rank2-display');
const suit1Display  = document.getElementById('suit1-display');
const suit2Display  = document.getElementById('suit2-display');
const situationText = document.getElementById('situation-text');
const gtoLine       = document.getElementById('gto-line');
const gtoBadge      = document.getElementById('gto-badge');
const pickerOverlay = document.getElementById('picker-overlay');
const pickerGrid    = document.getElementById('rank-grid');
const pickerTitle   = document.getElementById('picker-title');
const foldPct       = document.getElementById('fold-pct');
const callPct       = document.getElementById('call-pct');
const raisePct      = document.getElementById('raise-pct');
const callLabel     = document.getElementById('call-label');
const raiseLabel    = document.getElementById('raise-label');
const feedbackBar   = document.getElementById('quiz-feedback-bar');
const feedbackText  = document.getElementById('quiz-feedback-text');

// ---- Render cards ----
function renderCards() {
  const r1 = state.rank1, r2 = state.rank2;

  rank1Display.textContent = RANK_NAME[r1];
  rank2Display.textContent = RANK_NAME[r2];

  suit1Display.textContent = state.suit1;
  suit1Display.className = 'card-suit ' + (RED_SUITS.includes(state.suit1) ? 'red' : 'black');

  suit2Display.textContent = state.suit2;
  suit2Display.className = 'card-suit ' + (RED_SUITS.includes(state.suit2) ? 'red' : 'black');

  state.suited = (r1 !== r2) && (state.suit1 === state.suit2);
}

// ---- Render scenario UI ----
function renderScenario() {
  document.getElementById('opp-label').textContent    = OPP_LABELS[state.scenario];
  document.getElementById('player-label').textContent = MY_LABELS[state.scenario];
  situationText.textContent = SCENARIO_SITUATIONS[state.scenario];
  callLabel.textContent  = CALL_LABELS[state.scenario];
  raiseLabel.textContent = RAISE_LABELS[state.scenario];
}

// ---- Scenario buttons ----
document.querySelectorAll('.scenario-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.scenario = btn.dataset.scenario;
    renderScenario();
    if (state.mode === 'study') analyze();
  });
});

// ---- Mode buttons ----
document.getElementById('mode-study').addEventListener('click', () => switchMode('study'));
document.getElementById('mode-quiz').addEventListener('click',  () => switchMode('quiz'));

function switchMode(mode) {
  state.mode = mode;
  document.body.classList.toggle('quiz-mode', mode === 'quiz');
  document.getElementById('mode-study').classList.toggle('active', mode === 'study');
  document.getElementById('mode-quiz').classList.toggle('active',  mode === 'quiz');
  resetActionButtons();
  gtoLine.style.display = 'none';
  feedbackBar.style.display = 'none';
  foldPct.textContent = callPct.textContent = raisePct.textContent = '—';

  if (mode === 'study') {
    analyze();
  } else {
    loadQuiz();
  }
}

// ---- Card picker ----
document.getElementById('card1').addEventListener('click', () => openPicker(1));
document.getElementById('card2').addEventListener('click', () => openPicker(2));

function openPicker(slot) {
  if (state.mode === 'quiz') return;
  state.pickingSlot = slot;
  document.getElementById(`card${slot}`).classList.add('selected');
  showRankPicker();
  pickerOverlay.style.display = 'flex';
}

function showRankPicker() {
  state.pickingStage = 'rank';
  pickerTitle.textContent = 'Pick rank';
  pickerGrid.className = 'rank-grid';
  pickerGrid.innerHTML = '';
  RANKS.forEach(r => {
    const btn = document.createElement('button');
    btn.className = 'rank-pick';
    btn.textContent = RANK_NAME[r];
    btn.addEventListener('click', () => selectRank(r));
    pickerGrid.appendChild(btn);
  });
}

function selectRank(rank) {
  if (state.pickingSlot === 1) state.rank1 = rank;
  else state.rank2 = rank;

  // Keep rank1 >= rank2
  if (state.rank1 < state.rank2) {
    [state.rank1, state.rank2] = [state.rank2, state.rank1];
    [state.suit1, state.suit2] = [state.suit2, state.suit1];
  }

  showSuitPicker();
}

function showSuitPicker() {
  state.pickingStage = 'suit';
  pickerTitle.textContent = 'Pick suit';
  pickerGrid.className = 'suit-grid';
  pickerGrid.innerHTML = '';
  SUITS.forEach(s => {
    const btn = document.createElement('button');
    const isRed = RED_SUITS.includes(s);
    btn.className = 'suit-pick ' + (isRed ? 'red' : 'black');
    btn.textContent = s;
    btn.addEventListener('click', () => selectSuit(s));
    pickerGrid.appendChild(btn);
  });
}

function selectSuit(suit) {
  if (state.pickingSlot === 1) state.suit1 = suit;
  else state.suit2 = suit;

  document.getElementById('card1').classList.remove('selected');
  document.getElementById('card2').classList.remove('selected');
  pickerOverlay.style.display = 'none';
  state.pickingSlot = null;
  state.pickingStage = null;
  renderCards();
  if (state.mode === 'study') analyze();
}

pickerOverlay.addEventListener('click', e => {
  if (e.target === pickerOverlay) {
    pickerOverlay.style.display = 'none';
    document.getElementById('card1').classList.remove('selected');
    document.getElementById('card2').classList.remove('selected');
    state.pickingSlot = null;
    state.pickingStage = null;
  }
});

// ---- Analyze (study mode) ----
async function analyze() {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      rank1: state.rank1, rank2: state.rank2,
      suited: state.suited, scenario: state.scenario,
    }),
  });
  const data = await res.json();

  foldPct.textContent  = data.probs.fold  + '%';
  callPct.textContent  = data.probs.call  + '%';
  raisePct.textContent = data.probs.raise + '%';

  gtoBadge.textContent = data.gto_action;
  gtoBadge.className   = data.gto_action;
  gtoLine.style.display = 'flex';

  resetActionButtons();
  document.querySelectorAll('.action-btn').forEach(btn => {
    if (btn.dataset.action === data.gto_action) btn.classList.add('gto-action');
  });
}

// ---- Quiz mode ----
async function loadQuiz() {
  state.quizAnswered = false;
  state.quizAnswer   = null;
  feedbackBar.style.display = 'none';
  gtoLine.style.display = 'none';
  resetActionButtons();
  foldPct.textContent = callPct.textContent = raisePct.textContent = '—';

  const res  = await fetch(`/api/quiz?scenario=${state.scenario}`);
  const data = await res.json();

  state.quizAnswer = data.gto_action;
  state.rank1      = data.rank1;
  state.rank2      = data.rank2;
  state.suited     = data.suited;

  // Assign suits based on suited flag
  if (data.suited) {
    state.suit1 = '♥';
    state.suit2 = '♥';
  } else {
    state.suit1 = '♠';
    state.suit2 = '♥';
  }

  renderScenario();
  renderCards();
}

// ---- Quiz answer ----
document.querySelectorAll('.action-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (state.mode !== 'quiz' || state.quizAnswered) return;
    state.quizAnswered = true;

    const chosen  = btn.dataset.action;
    const correct = chosen === state.quizAnswer;

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rank1: state.rank1, rank2: state.rank2,
        suited: state.suited, scenario: state.scenario,
      }),
    });
    const data = await res.json();
    foldPct.textContent  = data.probs.fold  + '%';
    callPct.textContent  = data.probs.call  + '%';
    raisePct.textContent = data.probs.raise + '%';

    document.querySelectorAll('.action-btn').forEach(b => {
      if (b.dataset.action === state.quizAnswer) b.classList.add('correct-answer');
      else if (b === btn && !correct) b.classList.add('wrong-answer');
    });

    gtoBadge.textContent  = state.quizAnswer;
    gtoBadge.className    = state.quizAnswer;
    gtoLine.style.display = 'flex';

    feedbackText.textContent = correct ? '✓ Correct!' : `✗ Model says ${state.quizAnswer}`;
    feedbackText.className   = correct ? 'correct' : 'incorrect';
    feedbackBar.style.display = 'flex';
  });
});

document.getElementById('next-hand-btn').addEventListener('click', loadQuiz);

// ---- Helpers ----
function resetActionButtons() {
  document.querySelectorAll('.action-btn').forEach(b => {
    b.classList.remove('gto-action', 'correct-answer', 'wrong-answer');
  });
}

// ---- Init ----
renderCards();
renderScenario();
analyze();
