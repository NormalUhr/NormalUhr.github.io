// Door openings need to clear the assembled joinery, not only the walkable
// 3.60 m passage. Keep the masonry cuts and opened leaves on one contract.
export const GROUND_DOOR = Object.freeze({
  clearHalfWidth: 1.80,
  jambRadius: 33.045,
  jambDepth: .28,
  jambTangent: 1.90,
  jambWidth: .20,
  sillHalfWidth: 2.08,      // 80 mm beyond the oak jamb's outside face.
  baseCutHalfWidth: 2.22,   // 60 mm beyond the stone jamb foot.
  leafRearRadius: 33.225,   // 40 mm beyond the oak jamb's front face.
  leafLength: 1.72,
  leafTangent: 1.91,
  leafThickness: .13,
  openAngle: 75 * Math.PI / 180,
  hingeRadius: 33.205,
  hingeTangent: 1.817,
  sillInnerRadius: 32.73,
  sillOuterRadius: 33.27,
  jambFootTop: .01,         // Distinct from the gallery floor at Y = 0.
});
