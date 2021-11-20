import { MemoryRouter as Router, Switch, Route } from 'react-router-dom';
import './App.css';

const Hello = () => {
  return (
    <div>
      <h1>Neura Test One: Move a Fucking dot</h1>
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
