import { MemoryRouter as Router, Switch, Route } from 'react-router-dom';
import './App.css';

const Hello = () => { 
  return (
    <div>
        <h1>Neura Test One: Move a Fucking dot</h1>
        <div className="dot" style={{width: "25px", height: "25px", backgroundColor: "black"}}>
        </div>
    </div>
  );
};

export default function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" component={Hello} />
      </Switch>
    </Router>
  );
}
