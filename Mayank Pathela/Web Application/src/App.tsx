import React, { Component, useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Sector,
  Cell,
  ResponsiveContainer
} from "recharts";
import axios from "axios";
import Loader from "react-loader-spinner";

import "./App.scss";

const data = [
  {
    name: "Positive",
    value: 0.75,
    color: "#59c59f"
  },
  {
    name: "Negative",
    value: 0.25,
    color: "#ea4335"
  }
];

const getData = (sec: number) => {
  console.log("truggg");
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("Positive");

      console.log({ sentiment: "Positive" });
    }, sec);
  });
};

// In a while
const getSentiment = async (text: String) => {
  // const response = data;
  const response = await axios.post("http://127.0.0.1:5000/", { text });
  console.log(response); // yeah print and check
  const sentiment = response.data.sentiment;
  const res = data;
  res[0].value = sentiment;
  res[1].value = 1 - sentiment;

  return res;
};

const App = (props: any) => {
  const [activePieIndex, setActivePieIndex] = useState(0);
  const [input, setInput] = useState("");
  const [result, setResult] = useState([]);
  const [sentiment, setSentiment] = useState(null);
  const [timer, setTimer] = useState(null);

  useEffect(() => {
    async function fetchData(string: string) {
      // const data = await getSentiment(1000);

      const data = await getSentiment(input);
      console.log(data);
      const sentiment = data[0].value > data[1].value ? "Positive" : "Negative";
      setSentiment(sentiment);
      setResult(data);
    }
    if (input) {
      setResult("");
      clearTimeout(timer);
      setTimer(setTimeout(() => fetchData(input), 1000));
      return () => {
        setResult([]);
      };
    }
  }, [input]);

  return (
    <div className="App">
      <div className={`App-header ${input ? "App-header__active" : ""}`}>
        <form className="main-form">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            // type="textarea"
            className="form__field"
            placeholder="Enter text to analyze:"
          />
          {/* <button type="button" className="btn btn--primary btn--inside uppercase">Get</button> */}
        </form>
      </div>
      {input ? (
        result ? (
          <div className="result">
            <div className="result__images">
              <img
                src="./images/positive.png"
                className={`${sentiment === "Positive" ? "active" : ""}`}
                alt="positive"
              />
              <img
                src="./images/negative.png"
                className={`${sentiment === "Negative" ? "active" : ""}`}
                alt="negative"
              />
            </div>
            <div className="result__text">{sentiment}</div>

            <div className="sentiment-chart">
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    activeIndex={activePieIndex}
                    activeShape={renderActiveShape}
                    data={data}
                    cx={"50%"}
                    cy={175}
                    innerRadius={80}
                    nameKey="name"
                    dataKey="value"
                    outerRadius={120}
                    fill="#8884d8"
                    onMouseEnter={(a: any, b: any) =>
                      onPieEnter(a, b, setActivePieIndex)
                    }
                  >
                    {data.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                  <Legend wrapperStyle={{ position: "relative" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="loader-wrap">
            <Loader type="Plane" color="#00BFFF" height="100" width="100" />
          </div>
        )
      ) : (
        <></>
      )}
    </div>
  );
};

export default App;

const onPieEnter = (data: any, index: any, setPie: any): any => {
  setPie(index);
};

const renderActiveShape = (props: any) => {
  const RADIAN = Math.PI / 180;
  const {
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    startAngle,
    endAngle,
    fill,
    percent,
    value,
    name
  } = props;
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(-RADIAN * midAngle);
  const sx = cx + (outerRadius + 10) * cos;
  const sy = cy + (outerRadius + 10) * sin;
  const mx = cx + (outerRadius + 30) * cos;
  const my = cy + (outerRadius + 30) * sin;
  const ex = mx + (cos >= 0 ? 1 : -1) * 22;
  const ey = my;
  const textAnchor = cos >= 0 ? "start" : "end";

  return (
    <g>
      <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>
        {name}
      </text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 6}
        outerRadius={outerRadius + 10}
        fill={fill}
      />
      <path
        d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`}
        stroke={fill}
        fill="none"
      />
      <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
      <text
        x={ex + (cos >= 0 ? 1 : -1) * 12}
        y={ey}
        textAnchor={textAnchor}
        fill="white"
      >{`${name}: ${value}`}</text>
      <text
        x={ex + (cos >= 0 ? 1 : -1) * 12}
        y={ey}
        dy={18}
        textAnchor={textAnchor}
        fill="#999"
      >
        {`(${(percent * 100).toFixed(2)}%)`}
      </text>
    </g>
  );
};
