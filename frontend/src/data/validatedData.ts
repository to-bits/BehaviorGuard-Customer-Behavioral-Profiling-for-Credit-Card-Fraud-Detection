import type { Factor } from "../types";

export const validatedSummary = {
  datasetSize: 283726,
  normalTransactions: 283253,
  fraudTransactions: 473,
  fraudRate: 0.001667101358352777,
  threshold: 0.54,
  modelVersion: "xgboost-final-v1",
  modelName: "XGBoost fraud detector",
  xgboost: {
    precision: 0.9259259259259259,
    recall: 0.7894736842105263,
    f1: 0.8522727272727273,
    prAuc: 0.81050081582244,
    rocAuc: 0.9661874320051953,
    tn: 56645,
    fp: 6,
    fn: 20,
    tp: 75,
  },
  randomForest: { precision: 0.9459459459459459, recall: 0.7368421052631579, f1: 0.8284023668639053, prAuc: 0.8118040369615392, rocAuc: 0.9489328473785476 },
};

export const validatedHourlyFraud = [
  [0, 0.07846], [1, 0.23764], [2, 1.45103], [3, 0.48753], [4, 1.04356], [5, 0.36814],
  [6, 0.22048], [7, 0.31799], [8, 0.08796], [9, 0.10148], [10, 0.04834], [11, 0.31583],
  [12, 0.11055], [13, 0.11094], [14, 0.13923], [15, 0.15879], [16, 0.13418], [17, 0.17359],
  [18, 0.1651], [19, 0.12206], [20, 0.10775], [21, 0.09076], [22, 0.05853], [23, 0.15621],
].map(([hour, fraudRate]) => ({ hour: `${String(hour).padStart(2, "0")}:00`, fraudRate }));

export const validatedAmountCategories = [
  { category: "Zero", transactions: 1808, fraudCount: 25, fraudRate: 1.38274 },
  { category: "Low", transactions: 188340, fraudCount: 268, fraudRate: 0.1423 },
  { category: "Medium", transactions: 64808, fraudCount: 98, fraudRate: 0.15122 },
  { category: "High", transactions: 19661, fraudCount: 48, fraudRate: 0.24414 },
  { category: "Very High", transactions: 9109, fraudCount: 34, fraudRate: 0.37326 },
];

export const validatedBehavioralEvidence = [
  { feature: "Transactions_Last_1H", normalMedian: 7904, fraudMedian: 7774, adjustedP: 9.11446123222998e-9, cliffsDelta: -0.1543065336766407, effectSize: "Small" },
  { feature: "Transactions_Last_24H", normalMedian: 139787, fraudMedian: 113816, adjustedP: 5.3169103603296425e-8, cliffsDelta: -0.14607675345692528, effectSize: "Negligible" },
  { feature: "Amount_Log", normalMedian: 3.1354942159291497, fraudMedian: 2.381396273418336, adjustedP: 3.223308286040652e-5, cliffsDelta: -0.11154320394091988, effectSize: "Negligible" },
  { feature: "Amount_ZScore", normalMedian: -0.2654666003912112, fraudMedian: -0.3141088822601138, adjustedP: 3.223308286040652e-5, cliffsDelta: -0.11154320394091988, effectSize: "Negligible" },
];

export const validatedImportance: Factor[] = [
  ["V14", 0.31154808], ["V10", 0.1064477], ["V4", 0.06899252], ["Amount_ZScore", 0.06890216], ["V12", 0.058797408],
  ["Time_Period_Evening", 0.032244023], ["V8", 0.02883818], ["V20", 0.025519935], ["V3", 0.02532361], ["V19", 0.020719837],
  ["V17", 0.0168796], ["V13", 0.014939378], ["Transactions_Last_1H", 0.004280864],
].map(([feature, contribution]) => ({ feature: String(feature), contribution: Number(contribution), direction: "toward_fraud" as const, display_value: `mean model importance ${Number(contribution).toFixed(3)}` }));

export const demoTransactionPayload: Record<string, unknown> = {
  Time: 0, V1: -1.359807, V2: -0.072781, V3: 2.536347, V4: 1.378155, V5: -0.338321, V6: 0.462388,
  V7: 0.239599, V8: 0.098698, V9: 0.363787, V10: 0.090794, V11: -0.5516, V12: -0.617801, V13: -0.99139,
  V14: -0.311169, V15: 1.468177, V16: -0.470401, V17: 0.207971, V18: 0.025791, V19: 0.403993, V20: 0.251412,
  V21: -0.018307, V22: 0.277838, V23: -0.110474, V24: 0.066928, V25: 0.128539, V26: -0.189115, V27: 0.133558,
  V28: -0.021053, Amount: 149.62, Hour: 0, Time_Period: "Night", Amount_Log: 5.01476, Amount_Category: "Medium",
  Transactions_Last_1H: 0, Transactions_Last_24H: 0, Amount_ZScore: 0.244199,
};
