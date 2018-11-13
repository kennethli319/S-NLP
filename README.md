# Speech and Natural Language Processing
  This page shows some of the open source projects, toolkits and websites which are typically useful for researches or applications in speech and language processing.

---
## Speech related:
<ul id="id01">
  <li>
    <b>CMUSphinx</b> <br>
    OPEN SOURCE SPEECH RECOGNITION TOOLKIT <br>
    https://cmusphinx.github.io/
  </li>

  <li>
    <b>Festival</b> <br>
    Speech Synthesis System by CSTR <br>
    http://www.cstr.ed.ac.uk/projects/festival/
  </li>

  <li>
    <b>Speech-Corpus-Collection</b> <br> 
    https://github.com/candlewill/Speech-Corpus-Collection
  </li>

  <li>
    <b>Kaldi ASR</b> <br> 
    http://kaldi-asr.org/
  </li>

  <li>
    <b>Praat</b> <br> 
    http://www.fon.hum.uva.nl/praat/
  </li>

  <li>
    <b>LibriSpeech ASR corpus</b> <br>
    Large-scale (1000 hours) corpus of read English speech <br>
    https://www.openslr.org/12
  </li>

  <li>
    <b>Common voice</b> <br>
    Common Voice is Mozilla's initiative to help teach machines how real people speak. <br>
    https://github.com/mozilla/voice-web <br>
    https://voice.mozilla.org/
  </li>
  
  <li>
    <b>TED-LIUM Release 3</b> <br>
    452 hours of audio <br>
    https://www.openslr.org/51/
  </li>

  <li>
    <b>VoxForge</b> <br>
    VoxForge was set up to collect transcribed speech for use in Open Source Speech Recognition Engines ("SRE"s) such as such as ISIP, HTK, Julius and Sphinx.<br>
    http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/
  </li>
  
  <li>
    <b>Tatoeba</b> <br>
    Tatoeba is a collection of sentences and translations. <br>
    https://tatoeba.org/eng/downloads
  </li>
  
  <li>
    <b>EMIME Project</b> <br>
    http://www.emime.org/
  </li>
  
  <li>
    <b>CMU_ARCTIC speech synthesis databases</b> <br>
    http://festvox.org/cmu_arctic/
  </li>
  
  <li>
    <b>The World English Bible</b> <br>
    http://www.audiotreasure.com/webindex.htm
  </li>
  
  <li>
    <b>Nancy Corpus</b> <br>
    http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/
  </li>
  
  <li>
  <b>Google uis-rnn</b> <br>
  This is the library for the Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm, corresponding to the paper Fully Supervised Speaker Diarization. https://arxiv.org/abs/1810.04719
    https://github.com/google/uis-rnn
  </li>

  <li>
  <b>A simple interface for the CMU pronouncing dictionary </b> <br>
  https://github.com/aparrish/pronouncingpy
  </li>
  
  
  <li>
  <b>E-Guide dog</b> <br>
  https://www.oschina.net/question/3820517_2280583
  </li>
  
</ul>

## NLP related:
<ol>
  <li>
  <b>BBC news corpus</b> <br>
  http://mlg.ucd.ie/datasets/bbc.html
  </li>
  
  <li>
  <b>GEO query database</b> <br>
  http://www.cs.utexas.edu/users/ml/nldata/geoquery.html
  </li>
  
  <li>
  <b>FreeBase for QA</b> <br>
  https://www.freebase.com/
  </li>
  
  <li>
  <b>Google Bert </b> <br>
  TensorFlow code and pre-trained models for BERT https://arxiv.org/abs/1810.04805
  https://github.com/google-research/bert
  </li>
  
  https://www.dropbox.com/sh/fum6rxzx66bz5dl/AABLGonlZvj5m6LzqGgUsb6ga?dl=0
</ol>

## Machine Learning / Neural Network related:
 
<ol>
  <li>
  <b> An MIT Press book by Ian Goodfellow and Yoshua Bengio and Aaron Courville </b> <br>
  https://www.deeplearningbook.org/
  </li>
  
  <li>
  <b> Deepmind trfl </b> <br>
  https://github.com/deepmind/trfl/
  </li>
  
  <li>
  <b> Dopamine is a research framework for fast prototyping of reinforcement learning algorithms </b> <br>
  https://github.com/google/dopamine
  </li>

  <li>
  <b> Neural Networks and Deep Learning, a free online book </b> <br>
  http://neuralnetworksanddeeplearning.com/
  </li>
</ol>

<ul id="id01">
  <li>Oslo</li>
  <li>Stockholm</li>
  <li>Helsinki</li>
  <li>Berlin</li>
  <li>Rome</li>
  <li>Madrid</li>
</ul>

<script>
function sortList() {
  var list, i, switching, b, shouldSwitch;
  list = document.getElementById("id01");
  switching = true;
  /* Make a loop that will continue until
  no switching has been done: */
  while (switching) {
    // Start by saying: no switching is done:
    switching = false;
    b = list.getElementsByTagName("LI");
    // Loop through all list items:
    for (i = 0; i < (b.length - 1); i++) {
      // Start by saying there should be no switching:
      shouldSwitch = false;
      /* Check if the next item should
      switch place with the current item: */
      if (b[i].innerHTML.toLowerCase() > b[i + 1].innerHTML.toLowerCase()) {
        /* If next item is alphabetically lower than current item,
        mark as a switch and break the loop: */
        shouldSwitch = true;
        break;
      }
    }
    if (shouldSwitch) {
      /* If a switch has been marked, make the switch
      and mark the switch as done: */
      b[i].parentNode.insertBefore(b[i + 1], b[i]);
      switching = true;
    }
  }
}
</script>
