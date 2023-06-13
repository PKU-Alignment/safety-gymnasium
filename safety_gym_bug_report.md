# Safety-Gym Bug Report

## Natural Lidar

The original Natural Lidar in [Safe-Gym](https://github.com/openai/safety-gym) has a problem of not being able to detect low-lying objects, which may affect comprehensive environmental observations.

<table class="docutils align-default">
  <tbody>
    <tr class="row-odd">
      <td>
        <figure class="align-default">
          <img
              alt="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_high_lying_0.jpeg"
              src="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_high_lying_0.jpeg" style="width: 230px;">
        </figure>
        <p class="centered">
          <strong><a class="reference internal"><span class="std std-ref">Goal0</span></a></strong>
        </p>
      </td>
      <td>
        <figure class="align-default">
          <a class="reference external image-reference"><img
              alt="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_high_lying_1.jpeg"
              src="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_high_lying_1.jpeg" style="width: 230px;"></a>
        </figure>
        <p class="centered">
          <strong><a class="reference internal"><span class="std std-ref">Goal2</span></a></strong>
        </p>
      </td>
    </tr>
    <tr class="row-even">
      <td>
        <figure class="align-default">
          <a class="reference external image-reference"><img
              alt="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_low_lying_0.jpeg"
              src="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_low_lying_0.jpeg" style="width: 230px;"></a>
        </figure>
        <p class="centered">
          <strong><a class="reference internal"><span class="std std-ref">Button0</span></a></strong>
        </p>
      </td>
      <td>
        <figure class="align-default">
          <a class="reference external image-reference" href="./button#button2"><img
              alt="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_low_lying_1.jpeg"
              src="https://github.com/PKU-Alignment/safety-gymnasium/raw/HEAD/images/bug_report/detect_low_lying_1.jpeg" style="width: 230px;"></a>
        </figure>
        <p class="centered">
          <strong><a class="reference internal"><span class="std std-ref">Button2</span></a></strong>
        </p>
      </td>
    </tr>
  </tbody>
</table>
