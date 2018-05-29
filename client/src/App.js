import React, { Component } from 'react';
import logo from './bars.svg';
import './dropzone.css';
import './App.css';
import upload_icon from './upload.svg';
import play_icon from './play.svg';
import pause_icon from './pause.svg';
import axios from 'axios';
import settings from './settings.json';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      selected_sheet: "",
      uploader_container_className: "hidden",
      music_playing: false
    };
    this.onSelectedSheetChange = this.onSelectedSheetChange.bind(this);
    this.renderOsmd = this.renderOsmd.bind(this);
    this.resetOsmd = this.resetOsmd.bind(this);
    this.renderUploader = this.renderUploader.bind(this);
    this.toggleUploader = this.toggleUploader.bind(this);
    this.togglePlay = this.togglePlay.bind(this);
  }

  onSelectedSheetChange(event) {
    event.preventDefault();
    let sheet = event.target.value;
    this.setState({ selected_sheet: sheet });
    if (sheet === "") {
      this.resetOsmd();
      return;
    } else {
      axios.get(settings.endpoints.get_music_xml,
        {
          params: {
            sheet_name: sheet
          }
        }
      )
      .then(response => {
        let xml = response.data;
        if (xml !== "") {
          this.renderOsmd(xml);

        }
      });
    }
  }

  renderOsmd(xml) {
    this.resetOsmd();
    let osmd = new window.opensheetmusicdisplay.OpenSheetMusicDisplay("osmd", true, "svg");
    osmd.load(xml).then(
      function () {
        osmd.render();
      }
    );
  }

  toggleUploader() {
    let bool = (this.state.uploader_container_className === "open") ? "hidden" : "open";
    this.setState({ uploader_container_className: bool })
  }

  togglePlay() {
    console.log('a')
    let playing = (this.state.music_playing === true) ? false : true;
    this.setState({ music_playing: playing });

  }

  renderUploader() {
    console.log('Setting uploader url');
    let ctx = this;
    let to_wait = 1500;
    window.Dropzone.options.uploader = {
      url: settings.endpoints.image_upload,
      init: function () {
        this.on("success", function (file) {
          setTimeout(function () {
            ctx.toggleUploader();
          }, to_wait);
        });
      },
      clickable: false,
      maxFiles: 1
    };
  }

  resetOsmd() {
    window.document.querySelector('div[id="osmd"]').innerHTML = "";
  }

  componentDidMount() {
    this.renderUploader();
  }

  render() {
    return (
      <div className="App">

        <div className="header">
          <div className="left">
            <img src={logo} className="logo" alt="logo" />
            <div className="title">
              Musically
            </div>
            <div className="round-btn" onClick={this.toggleUploader}>
              <div className="round-btn-inside">
                <img src={upload_icon} className="upload-svg" alt="Upload files" />
              </div>
            </div>
            <div className="play-btn round-btn" onClick={this.togglePlay}>
              <div className="round-btn-inside">
                <img src={this.state.music_playing === true ? pause_icon : play_icon} className="play-svg" alt="Play the music" />
              </div>
            </div>
          </div>

          <div className="right">
            <div className="credits">
              <span>By Angus, Annie and Yiwei</span>
            </div>
          </div>
        </div>

        <div className={"dropzone-container " + this.state.uploader_container_className} onClick={this.toggleUploader}>
          <div id="uploader" className="dropzone"></div>
        </div>

        <br />

        <div style={{ fontSize: 12 + 'pt' }}>Sample sheet:</div>

        <select name="selected-sheet"
          onChange={this.onSelectedSheetChange}
          value={this.state.selected_sheet}>
          <option value=""></option>
          <option value="ActorPreludeSample">Chant</option>
          <option value="BeetAnGeSample">BeetAnGeSample</option>
          <option value="HelloWorld">HelloWorld</option>
          <option value="MozartPianoSonata">MozartPianoSonata</option>
        </select>

        <div id="osmd"></div>

      </div>
    );
  }
}

export default App;
