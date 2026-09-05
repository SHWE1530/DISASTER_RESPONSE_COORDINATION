// ============================================================
// DISASTER RESPONSE AI
// FRONTEND JAVASCRIPT
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const predictButton =
    document.getElementById(
        "predictButton"
    );

const backendStatus =
    document.getElementById(
        "backendStatus"
    );

const emptyState =
    document.getElementById(
        "emptyState"
    );

const result =
    document.getElementById(
        "result"
    );

const errorBox =
    document.getElementById(
        "errorBox"
    );

const riskBadge =
    document.getElementById(
        "riskBadge"
    );


// ============================================================
// GET VALUE
// ============================================================

function value(id) {

    return document
        .getElementById(id)
        .value;
}


function numberValue(id) {

    return Number(
        document
            .getElementById(id)
            .value
    );
}


// ============================================================
// BACKEND STATUS
// ============================================================

async function checkBackend() {

    try {

        const response =
            await fetch("/health");

        if (!response.ok) {

            throw new Error();
        }

        backendStatus.textContent =
            "● Backend Online";

        backendStatus.className =
            "status online";

    }

    catch {

        backendStatus.textContent =
            "● Backend Offline";

        backendStatus.className =
            "status offline";
    }
}


// ============================================================
// CREATE PAYLOAD
// ============================================================

function createPayload() {

    return {

        timestamp:
            value("timestamp"),

        state:
            value("state"),

        district:
            value("district"),

        rainfall_mm:
            numberValue("rainfall_mm"),

        river_level_m:
            numberValue("river_level_m"),

        river_level_threshold_m:
            numberValue(
                "river_level_threshold_m"
            ),

        emergency_calls:
            numberValue(
                "emergency_calls"
            ),

        road_closures:
            numberValue(
                "road_closures"
            ),

        bridge_closures:
            numberValue(
                "bridge_closures"
            ),

        flood_history_count:
            numberValue(
                "flood_history_count"
            ),

        population_affected:
            numberValue(
                "population_affected"
            ),

        water_level_change_m:
            numberValue(
                "water_level_change_m"
            )
    };
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    const risk =
        String(
            data.risk_level
        ).toLowerCase();


    emptyState.classList.add(
        "hidden"
    );

    result.classList.remove(
        "hidden"
    );

    errorBox.classList.add(
        "hidden"
    );


    // Risk badge

    riskBadge.textContent =
        risk.toUpperCase();

    riskBadge.className =
        "risk-badge";


    if (risk === "severe") {

        riskBadge.classList.add(
            "risk-severe"
        );

    }

    else if (risk === "moderate") {

        riskBadge.classList.add(
            "risk-moderate"
        );

    }

    else {

        riskBadge.classList.add(
            "risk-low"
        );
    }


    // District

    document.getElementById(
        "resultDistrict"
    ).textContent =
        data.district;


    // Confidence

    if (
        data.confidence !== null &&
        data.confidence !== undefined
    ) {

        document.getElementById(
            "confidence"
        ).textContent =
            `${(
                data.confidence * 100
            ).toFixed(2)}%`;

    }

    else {

        document.getElementById(
            "confidence"
        ).textContent =
            "N/A";
    }


    // Details

    document.getElementById(
        "resultRainfall"
    ).textContent =
        `${value("rainfall_mm")} mm`;


    document.getElementById(
        "resultRiver"
    ).textContent =
        `${value("river_level_m")} m`;


    document.getElementById(
        "resultCalls"
    ).textContent =
        value("emergency_calls");


    document.getElementById(
        "resultRoads"
    ).textContent =
        value("road_closures");


    document.getElementById(
        "resultBridges"
    ).textContent =
        value("bridge_closures");
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    errorBox.textContent =
        "❌ " + message;

    errorBox.classList.remove(
        "hidden"
    );

    result.classList.add(
        "hidden"
    );
}


// ============================================================
// PREDICT
// ============================================================

async function predictRisk() {

    predictButton.disabled =
        true;

    predictButton.textContent =
        "⏳ ANALYZING...";


    try {

        const payload =
            createPayload();


        const response =
            await fetch(
                "/predict-risk",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            payload
                        )
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }


        displayResult(data);

    }

    catch (error) {

        showError(
            error.message
        );
    }

    finally {

        predictButton.disabled =
            false;

        predictButton.textContent =
            "🚨 ANALYZE ZONE RISK";
    }
}


// ============================================================
// BUTTON
// ============================================================

predictButton.addEventListener(
    "click",
    predictRisk
);


// ============================================================
// INITIAL CHECK
// ============================================================

checkBackend();


// Check backend every 10 seconds

setInterval(
    checkBackend,
    10000
);