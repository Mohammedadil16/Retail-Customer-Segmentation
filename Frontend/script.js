// ============================================================
// RETAIL CUSTOMER SEGMENTATION
// FRONTEND JAVASCRIPT
// ============================================================


// Flask and frontend are on the same server
const API_URL = "";


// ============================================================
// PREDICT CUSTOMER
// ============================================================

async function predictCustomer() {

    const orders =
        document.getElementById(
            "numberOfOrders"
        ).value;

    const quantity =
        document.getElementById(
            "totalQuantityInput"
        ).value;

    const spending =
        document.getElementById(
            "totalSpendingInput"
        ).value;

    const average =
        document.getElementById(
            "averageOrderValueInput"
        ).value;


    // --------------------------------------------------------
    // VALIDATION
    // --------------------------------------------------------

    if (
        orders === "" ||
        quantity === "" ||
        spending === "" ||
        average === ""
    ) {

        alert(
            "Please enter all customer purchase details."
        );

        return;

    }


    if (
        Number(orders) <= 0 ||
        Number(quantity) <= 0 ||
        Number(spending) <= 0 ||
        Number(average) <= 0
    ) {

        alert(
            "Please enter values greater than zero."
        );

        return;

    }


    // --------------------------------------------------------
    // SHOW LOADING
    // --------------------------------------------------------

    document.getElementById(
        "loading"
    ).style.display = "block";


    document.getElementById(
        "results"
    ).style.opacity = "0.5";


    try {


        // ----------------------------------------------------
        // SEND REQUEST TO FLASK
        // ----------------------------------------------------

        const response = await fetch(
            `${API_URL}/predict`,
            {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    number_of_orders:
                        Number(orders),

                    total_quantity:
                        Number(quantity),

                    total_spending:
                        Number(spending),

                    average_order_value:
                        Number(average)

                })

            }
        );


        const result =
            await response.json();


        // ----------------------------------------------------
        // ERROR
        // ----------------------------------------------------

        if (!response.ok || !result.success) {

            throw new Error(
                result.error ||
                "Prediction failed."
            );

        }


        // ----------------------------------------------------
        // PREDICTIONS
        // ----------------------------------------------------

        const prediction =
            result.prediction;


        document.getElementById(
            "kmeansResult"
        ).innerHTML =

            `${prediction.kmeans_segment}
            <br>
            <span class="cluster-small">
            (${prediction.kmeans})
            </span>`;


        document.getElementById(
            "hierarchicalResult"
        ).innerHTML =

            `${prediction.hierarchical_segment}
            <br>
            <span class="cluster-small">
            (${prediction.hierarchical})
            </span>`;


        document.getElementById(
            "dbscanResult"
        ).innerHTML =

            `${prediction.dbscan_segment}
            <br>
            <span class="cluster-small">
            (${prediction.dbscan})
            </span>`;


        // ----------------------------------------------------
        // FINAL SEGMENT
        // ----------------------------------------------------

        document.getElementById(
            "customerType"
        ).textContent =
            prediction.customer_type;


        // ----------------------------------------------------
        // CUSTOMER DETAILS
        // ----------------------------------------------------

        document.getElementById(
            "displayOrders"
        ).textContent =
            formatNumber(
                result.input.number_of_orders
            );


        document.getElementById(
            "displayQuantity"
        ).textContent =
            formatNumber(
                result.input.total_quantity
            );


        document.getElementById(
            "displaySpending"
        ).textContent =
            formatCurrency(
                result.input.total_spending
            );


        document.getElementById(
            "displayAverage"
        ).textContent =
            formatCurrency(
                result.input.average_order_value
            );


        // ----------------------------------------------------
        // SCROLL TO RESULTS
        // ----------------------------------------------------

        document.getElementById(
            "results"
        ).scrollIntoView({

            behavior: "smooth",

            block: "start"

        });


    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            "Unable to connect to the Flask server.\n\n" +
            "Make sure Backend/app.py is running."
        );

    }

    finally {

        document.getElementById(
            "loading"
        ).style.display = "none";


        document.getElementById(
            "results"
        ).style.opacity = "1";

    }

}


// ============================================================
// CLEAR FORM
// ============================================================

function clearForm() {

    document.getElementById(
        "numberOfOrders"
    ).value = "";


    document.getElementById(
        "totalQuantityInput"
    ).value = "";


    document.getElementById(
        "totalSpendingInput"
    ).value = "";


    document.getElementById(
        "averageOrderValueInput"
    ).value = "";


    document.getElementById(
        "kmeansResult"
    ).textContent = "-";


    document.getElementById(
        "hierarchicalResult"
    ).textContent = "-";


    document.getElementById(
        "dbscanResult"
    ).textContent = "-";


    document.getElementById(
        "customerType"
    ).textContent = "-";


    document.getElementById(
        "displayOrders"
    ).textContent = "-";


    document.getElementById(
        "displayQuantity"
    ).textContent = "-";


    document.getElementById(
        "displaySpending"
    ).textContent = "-";


    document.getElementById(
        "displayAverage"
    ).textContent = "-";

}


// ============================================================
// NUMBER FORMAT
// ============================================================

function formatNumber(value) {

    return Number(value).toLocaleString(
        "en-IN"
    );

}


// ============================================================
// CURRENCY FORMAT
// ============================================================

function formatCurrency(value) {

    return Number(value).toLocaleString(
        "en-IN",
        {

            style: "currency",

            currency: "INR",

            minimumFractionDigits: 2

        }
    );

}


// ============================================================
// CHART VARIABLES
// ============================================================

let kmeansChart = null;

let hierarchicalChart = null;

let dbscanChart = null;

let scoreChart = null;


// ============================================================
// CREATE BAR CHART
// ============================================================

function createBarChart(
    canvasId,
    labels,
    values,
    title
) {

    const canvas =
        document.getElementById(
            canvasId
        );


    if (!canvas) {

        return null;

    }


    return new Chart(
        canvas,
        {

            type: "bar",

            data: {

                labels: labels,

                datasets: [

                    {

                        label: title,

                        data: values,

                        backgroundColor:
                            "#326c9f",

                        borderColor:
                            "#45b8ff",

                        borderWidth: 2,

                        borderRadius: 10

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        display: false

                    }

                },

                scales: {

                    x: {

                        ticks: {

                            color: "#8fa4c4"

                        },

                        grid: {

                            color:
                                "rgba(255,255,255,0.04)"

                        }

                    },

                    y: {

                        beginAtZero: true,

                        ticks: {

                            color: "#8fa4c4"

                        },

                        grid: {

                            color:
                                "rgba(255,255,255,0.04)"

                        }

                    }

                }

            }

        }

    );

}


// ============================================================
// CREATE DBSCAN CHART
// ============================================================

function createDBSCANChart(
    labels,
    values
) {

    const canvas =
        document.getElementById(
            "dbscanChart"
        );


    if (!canvas) {

        return null;

    }


    return new Chart(
        canvas,
        {

            type: "doughnut",

            data: {

                labels: labels,

                datasets: [

                    {

                        data: values,

                        backgroundColor: [

                            "#36a2eb",

                            "#ff6384",

                            "#9966ff",

                            "#ffcd56"

                        ],

                        borderColor:
                            "#ffffff",

                        borderWidth: 2

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        labels: {

                            color: "#aebed7"

                        }

                    }

                }

            }

        }

    );

}


// ============================================================
// CREATE SCORE CHART
// ============================================================

function createScoreChart(
    labels,
    values
) {

    const canvas =
        document.getElementById(
            "scoreChart"
        );


    if (!canvas) {

        return null;

    }


    return new Chart(
        canvas,
        {

            type: "bar",

            data: {

                labels: labels,

                datasets: [

                    {

                        label:
                            "Silhouette Score",

                        data: values,

                        backgroundColor:
                            "#326c9f",

                        borderColor:
                            "#45b8ff",

                        borderWidth: 2,

                        borderRadius: 10

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                scales: {

                    x: {

                        ticks: {

                            color:
                                "#8fa4c4"

                        }

                    },

                    y: {

                        beginAtZero: true,

                        max: 1,

                        ticks: {

                            color:
                                "#8fa4c4"

                        }

                    }

                },

                plugins: {

                    legend: {

                        display: false

                    }

                }

            }

        }

    );

}


// ============================================================
// LOAD DASHBOARD
// ============================================================

async function loadDashboard() {

    try {


        const response =
            await fetch(
                `${API_URL}/dashboard`
            );


        const data =
            await response.json();


        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Dashboard loading failed."
            );

        }


        // ----------------------------------------------------
        // STATISTICS
        // ----------------------------------------------------

        const stats =
            data.statistics;


        document.getElementById(
            "dashboardTotalCustomers"
        ).textContent =
            formatNumber(
                stats.total_customers
            );


        document.getElementById(
            "dashboardTotalOrders"
        ).textContent =
            formatNumber(
                stats.total_orders
            );


        document.getElementById(
            "dashboardTotalQuantity"
        ).textContent =
            formatNumber(
                stats.total_quantity
            );


        document.getElementById(
            "dashboardTotalSpending"
        ).textContent =
            formatCurrency(
                stats.total_spending
            );


        document.getElementById(
            "dashboardAverageOrderValue"
        ).textContent =
            formatCurrency(
                stats.average_order_value
            );


        // ----------------------------------------------------
        // DESTROY OLD CHARTS
        // ----------------------------------------------------

        if (kmeansChart) {

            kmeansChart.destroy();

        }


        if (hierarchicalChart) {

            hierarchicalChart.destroy();

        }


        if (dbscanChart) {

            dbscanChart.destroy();

        }


        if (scoreChart) {

            scoreChart.destroy();

        }


        // ----------------------------------------------------
        // K-MEANS CHART
        // ----------------------------------------------------

        kmeansChart =
            createBarChart(

                "kmeansChart",

                Object.keys(
                    data.kmeans
                ),

                Object.values(
                    data.kmeans
                ),

                "Customers"

            );


        // ----------------------------------------------------
        // HIERARCHICAL CHART
        // ----------------------------------------------------

        hierarchicalChart =
            createBarChart(

                "hierarchicalChart",

                Object.keys(
                    data.hierarchical
                ),

                Object.values(
                    data.hierarchical
                ),

                "Customers"

            );


        // ----------------------------------------------------
        // DBSCAN CHART
        // ----------------------------------------------------

        dbscanChart =
            createDBSCANChart(

                Object.keys(
                    data.dbscan
                ),

                Object.values(
                    data.dbscan
                )

            );


        // ----------------------------------------------------
        // PERFORMANCE
        // ----------------------------------------------------

        scoreChart =
            createScoreChart(

                Object.keys(
                    data.silhouette_scores
                ),

                Object.values(
                    data.silhouette_scores
                )

            );


    }

    catch (error) {

        console.error(
            "Dashboard error:",
            error
        );

    }

}


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadDashboard();

    }
);