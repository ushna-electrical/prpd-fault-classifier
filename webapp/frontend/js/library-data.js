/* Fault Library content for the 10 classes the deployed Sensor-2 model predicts.
 *
 * Every field is taken verbatim (or as a verbatim excerpt) from the final thesis,
 * "AI-Based Evaluation of PRPD Patterns for Medium Voltage Cables" (University of
 * Stuttgart), chapters/04_setup.tex:
 *   - summary, defectFamily, cables, recordings: Table 4.3 "Detail of faults used
 *     in this thesis" (\label{tab:setup:faults})
 *   - detail paragraphs: the per-fault prose of Sections 4.7 "Grounding-Related
 *     Faults", 4.8 "Artificial Air Gaps" and 4.9 "Artificial Creepage Path".
 * Nothing here is paraphrased or invented; wording not in the thesis is not used.
 * fault_2 absorbs the former fault_3 folder (ch. 8: "the fault_3 folder ... belong
 * to F2"), so its images include both.
 *
 * Images: one per sensor from the curated Table 5.5 set (served by the backend
 * at /library-images/<filename>).
 */
(function (global) {
  "use strict";

  var GROUNDING_INTRO = "The grounding faults investigate how PD behaves, and what its pattern looks like, " +
    "when a cable or part of it is not properly grounded.";

  var AIR_GAP_INTRO = "An artificial air gap reproduces a local discontinuity between insulation and " +
    "stress-control components that should be in contact. Since air has a much lower permittivity than the " +
    "surrounding solid insulation, the field redistributes and a locally increased stress develops within the " +
    "gap and along the adjacent interface. If the local field exceeds the inception level, discharges occur in " +
    "the gap or along the neighbouring interface.";

  var AIR_GAP_OUTRO = "The shape of the pattern looked different for every position.";

  var LIBRARY = [
    {
      code: "fault_2",
      display: "F2",
      title: "Cable Termination Outer-Surface Grounding Disconnection",
      defectFamily: "Grounding",
      cables: "01, 02, 04",
      recordings: "49",
      detail: [
        GROUNDING_INTRO,
        "In normal operation the outer conductive surface of the termination is kept at earth potential by a " +
        "separate grounding lead at its rear. To reproduce a missing or incorrectly installed connection, this " +
        "lead was disconnected while all other cable and test-system grounds were left unchanged. The outer " +
        "surface of the termination is then no longer referenced to earth. This changes the field distribution " +
        "around the termination, particularly along its external insulating surface, and the increased " +
        "tangential stress can promote surface discharges that are detected in the PRPD measurement. For this " +
        "defect all three sensors showed similar results."
      ],
      images: [
        { sensor: 1, file: "fault 02 sensor 1.png" },
        { sensor: 2, file: "fault 02 sensor 2.png" },
        { sensor: 3, file: "Fault 02 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(a)"
    },
    {
      code: "fault_4",
      display: "F4",
      title: "Cable Metallic Screen–Earth Disconnection",
      defectFamily: "Grounding",
      cables: "01",
      recordings: "32",
      detail: [
        GROUNDING_INTRO,
        "The metallic screen of an MV cable is normally earthed so that it stays at a defined potential and " +
        "provides a controlled return path for capacitive and high-frequency currents. Here the screen earth " +
        "was intentionally disconnected. This was only possible on Cable 01; on the other cables the screen " +
        "remains grounded through the other end, so no defect results. Removing the screen earth changes the " +
        "electrical boundary conditions: depending on the remaining grounding, the screen may float or be " +
        "referenced to earth only capacitively or through another point. This modifies the local field and the " +
        "current return path around the termination or joint and can raise the stress in normally " +
        "field-controlled regions. The fault reproduces a practical installation error, a missing or wrongly " +
        "connected screen bond, and is reversible. The pattern shape was similar on all three sensors, and most " +
        "points were detected."
      ],
      images: [
        { sensor: 1, file: "fault 04 sensor 01.png" },
        { sensor: 2, file: "fault 04 sensor 02.png" },
        { sensor: 3, file: "fault 04 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(b)"
    },
    {
      code: "fault_5",
      display: "F5",
      title: "Artificial Air-Gap in the Cable Joint",
      defectFamily: "Air gap",
      cables: "01, 04",
      recordings: "41 (+7 for the F5i variant, superimposed on Cable 01's internal PD)",
      detail: [
        AIR_GAP_INTRO,
        "In the cable joint (F5) about 10–11 cm of a cable tie was inserted into the joint, mainly on " +
        "Cable 01, whose joint had to be opened mechanically for this; it represents a joint that was not " +
        "properly installed and later covered with insulation layers. Because Cable 01 carries an internal " +
        "discharge, these recordings contain the air gap superimposed on the internal PD.",
        AIR_GAP_OUTRO
      ],
      images: [
        { sensor: 1, file: "fault 05 sensor 01.png" },
        { sensor: 2, file: "fault 05 sensor 02.png" },
        { sensor: 3, file: "fault 05 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.8"
    },
    {
      code: "fault_6",
      display: "F6",
      title: "Artificial Air-Gap in the Cable Head / Termination Interface",
      defectFamily: "Air gap",
      cables: "01, 02, 04",
      recordings: "46",
      detail: [
        AIR_GAP_INTRO,
        "At the cable head / termination interface (F6) a cable tie was inserted at the top head of the cable.",
        AIR_GAP_OUTRO
      ],
      images: [
        { sensor: 1, file: "fault 06 sensor 01.png" },
        { sensor: 2, file: "fault 06 sensor 02.png" },
        { sensor: 3, file: "fault 06 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.8"
    },
    {
      code: "fault_7",
      display: "F7",
      title: "Artificial Creepage Path from Cable Ring to Ground",
      defectFamily: "Creepage",
      cables: "01, 02, 04",
      recordings: "56 (one or two wooden blocks)",
      detail: [
        "An artificial creepage path was introduced at the cable-ring region to reproduce an external " +
        "surface-discharge condition between the metallic cable connection and earth. The cable ring was placed " +
        "on an insulating wooden support above a grounded conductive plate, creating a defined surface path from " +
        "the cable connection towards ground. Different heights of 12, 18 and 21 cm were used by stacking " +
        "wooden blocks (F7 with one block, F7A with two). Under normal conditions insulation geometry and " +
        "clearances prevent significant leakage currents and surface discharges. The wooden support redistributes " +
        "the field along the insulating surface, and at sufficiently high stress leakage current develops along " +
        "the wood and discharges occur along the creepage path. The configuration represents installations in " +
        "which an energised or field-stressed part is too close to grounded structures across an insulating " +
        "surface."
      ],
      images: [
        { sensor: 1, file: "fault 07 sensor 01.png" },
        { sensor: 2, file: "fault 07 sensor 02.png" },
        { sensor: 3, file: "fault 07 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.9"
    },
    {
      code: "fault_8",
      display: "F8",
      title: "Combined Termination and Cable-Screen Grounding Disconnection",
      defectFamily: "Grounding",
      cables: "01",
      recordings: "44",
      detail: [
        GROUNDING_INTRO,
        "A more severe grounding deficiency was created by disconnecting the earth of the termination outer " +
        "surface and of the metallic screen at the same time, the combination of F2 and F4, again only possible " +
        "on Cable 01 because of the joint. With both grounding paths removed, the affected components may take " +
        "floating or capacitively coupled potentials, a larger part of the accessory is subjected to undefined " +
        "potentials, and discharges may appear in regions with strong field gradients or along insulating " +
        "interfaces next to the disconnected surfaces. When the two faults were introduced separately, a UHF " +
        "pattern was visible. With both grounds disconnected together, the UHF sensor showed no pattern at all."
      ],
      images: [
        { sensor: 1, file: "fault 08 sensor 01.png" },
        { sensor: 2, file: "fault 08 sensor 02.png" },
        { sensor: 3, file: "fault 08 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(c)"
    },
    {
      code: "fault_9",
      display: "F9",
      title: "Far-End Cable Screen Ground Disconnection",
      defectFamily: "Grounding",
      cables: "01",
      recordings: "45",
      detail: [
        GROUNDING_INTRO,
        "Here the screen earth at the remote end of Cable 01 was removed while the local grounding of the " +
        "investigated termination was left unchanged. The screen then operates single-end grounded instead of " +
        "grounded at both ends, which changes the screen potential distribution and the path for capacitive and " +
        "high-frequency currents along the cable."
      ],
      images: [
        { sensor: 1, file: "fault 09 sensor 01.png" },
        { sensor: 2, file: "fault 09 sensor 02.png" },
        { sensor: 3, file: "fault 09 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(d)"
    },
    {
      code: "fault_10",
      display: "F10",
      title: "Complete Cable and Termination Grounding Disconnection",
      defectFamily: "Grounding",
      cables: "01, 02, 04",
      recordings: "32",
      detail: [
        GROUNDING_INTRO,
        "The local grounding connections of the metallic screen and of the termination, both at the top/termination " +
        "region and at the far end, were disconnected. This reproduces a severe loss of the intended grounding " +
        "over the whole cable-accessory system, for example when several earth connections are missing or " +
        "wrongly installed. Without a direct earth reference, conductive and semiconductive parts may float or " +
        "couple capacitively, the field distribution around the termination and screen deviates strongly from " +
        "normal, and discharges with characteristics of floating potentials or surface discharges can be expected."
      ],
      images: [
        { sensor: 1, file: "fault 10 sensor 01.png" },
        { sensor: 2, file: "fault 10 sensor 02.png" },
        { sensor: 3, file: "fault 10 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(e)"
    },
    {
      code: "fault_12",
      display: "F12",
      title: "Artificial Air-Gap in the Cable Cone",
      defectFamily: "Air gap",
      cables: "01, 02",
      recordings: "25",
      detail: [
        AIR_GAP_INTRO,
        "In the cable cone (F12) a cable tie was inserted into the cone; a buzzing sound was noted during these " +
        "tests.",
        AIR_GAP_OUTRO
      ],
      images: [
        { sensor: 1, file: "fault 12 sensor 01.png" },
        { sensor: 2, file: "fault 12 sensor 02.png" },
        { sensor: 3, file: "fault 12 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.8"
    },
    {
      code: "fault_13",
      display: "F13",
      title: "Connection-Box Housing Grounding Disconnection",
      defectFamily: "Grounding",
      cables: "01, 02, 04",
      recordings: "36",
      detail: [
        GROUNDING_INTRO,
        "In this fault configuration, the protective earth connection of the metallic connection-box housing was " +
        "intentionally removed. As a result, the housing was no longer maintained at a defined earth potential " +
        "and could assume a floating potential through capacitive coupling with the energized parts of the test " +
        "setup.",
        "The loss of the earth reference modifies the local electric-field distribution around the enclosure and " +
        "adjacent insulation surfaces. If the resulting potential difference becomes sufficiently large, " +
        "partial-discharge activity may develop at regions with enhanced electric-field stress, such as edges, " +
        "gaps, or nearby insulating interfaces. The corresponding PRPD behaviour can therefore exhibit " +
        "characteristics associated with floating-potential or surface-discharge activity."
      ],
      images: [
        { sensor: 1, file: "fault 13 sensor 01.png" },
        { sensor: 2, file: "fault 13 sensor 02.png" },
        { sensor: 3, file: "fault 13 sensor 03.png" }
      ],
      source: "thesis chapters/04_setup.tex, Table 4.3 and §4.7(f)"
    }
  ];

  global.PrpdLibrary = LIBRARY;
})(window);
